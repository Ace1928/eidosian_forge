import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def f2py_sources(self, sources, extension):
    new_sources = []
    f2py_sources = []
    f_sources = []
    f2py_targets = {}
    target_dirs = []
    ext_name = extension.name.split('.')[-1]
    skip_f2py = 0
    for source in sources:
        base, ext = os.path.splitext(source)
        if ext == '.pyf':
            if self.inplace:
                target_dir = os.path.dirname(base)
            else:
                target_dir = appendpath(self.build_src, os.path.dirname(base))
            if os.path.isfile(source):
                name = get_f2py_modulename(source)
                if name != ext_name:
                    raise DistutilsSetupError('mismatch of extension names: %s provides %r but expected %r' % (source, name, ext_name))
                target_file = os.path.join(target_dir, name + 'module.c')
            else:
                log.debug("  source %s does not exist: skipping f2py'ing." % source)
                name = ext_name
                skip_f2py = 1
                target_file = os.path.join(target_dir, name + 'module.c')
                if not os.path.isfile(target_file):
                    log.warn('  target %s does not exist:\n   Assuming %smodule.c was generated with "build_src --inplace" command.' % (target_file, name))
                    target_dir = os.path.dirname(base)
                    target_file = os.path.join(target_dir, name + 'module.c')
                    if not os.path.isfile(target_file):
                        raise DistutilsSetupError('%r missing' % (target_file,))
                    log.info('   Yes! Using %r as up-to-date target.' % target_file)
            target_dirs.append(target_dir)
            f2py_sources.append(source)
            f2py_targets[source] = target_file
            new_sources.append(target_file)
        elif fortran_ext_match(ext):
            f_sources.append(source)
        else:
            new_sources.append(source)
    if not (f2py_sources or f_sources):
        return new_sources
    for d in target_dirs:
        self.mkpath(d)
    f2py_options = extension.f2py_options + self.f2py_opts
    if self.distribution.libraries:
        for name, build_info in self.distribution.libraries:
            if name in extension.libraries:
                f2py_options.extend(build_info.get('f2py_options', []))
    log.info('f2py options: %s' % f2py_options)
    if f2py_sources:
        if len(f2py_sources) != 1:
            raise DistutilsSetupError('only one .pyf file is allowed per extension module but got more: %r' % (f2py_sources,))
        source = f2py_sources[0]
        target_file = f2py_targets[source]
        target_dir = os.path.dirname(target_file) or '.'
        depends = [source] + extension.depends
        if (self.force or newer_group(depends, target_file, 'newer')) and (not skip_f2py):
            log.info('f2py: %s' % source)
            from numpy.f2py import f2py2e
            f2py2e.run_main(f2py_options + ['--build-dir', target_dir, source])
        else:
            log.debug("  skipping '%s' f2py interface (up-to-date)" % source)
    else:
        if is_sequence(extension):
            name = extension[0]
        else:
            name = extension.name
        target_dir = os.path.join(*[self.build_src] + name.split('.')[:-1])
        target_file = os.path.join(target_dir, ext_name + 'module.c')
        new_sources.append(target_file)
        depends = f_sources + extension.depends
        if (self.force or newer_group(depends, target_file, 'newer')) and (not skip_f2py):
            log.info('f2py:> %s' % target_file)
            self.mkpath(target_dir)
            from numpy.f2py import f2py2e
            f2py2e.run_main(f2py_options + ['--lower', '--build-dir', target_dir] + ['-m', ext_name] + f_sources)
        else:
            log.debug("  skipping f2py fortran files for '%s' (up-to-date)" % target_file)
    if not os.path.isfile(target_file):
        raise DistutilsError('f2py target file %r not generated' % (target_file,))
    build_dir = os.path.join(self.build_src, target_dir)
    target_c = os.path.join(build_dir, 'fortranobject.c')
    target_h = os.path.join(build_dir, 'fortranobject.h')
    log.info("  adding '%s' to sources." % target_c)
    new_sources.append(target_c)
    if build_dir not in extension.include_dirs:
        log.info("  adding '%s' to include_dirs." % build_dir)
        extension.include_dirs.append(build_dir)
    if not skip_f2py:
        import numpy.f2py
        d = os.path.dirname(numpy.f2py.__file__)
        source_c = os.path.join(d, 'src', 'fortranobject.c')
        source_h = os.path.join(d, 'src', 'fortranobject.h')
        if newer(source_c, target_c) or newer(source_h, target_h):
            self.mkpath(os.path.dirname(target_c))
            self.copy_file(source_c, target_c)
            self.copy_file(source_h, target_h)
    else:
        if not os.path.isfile(target_c):
            raise DistutilsSetupError('f2py target_c file %r not found' % (target_c,))
        if not os.path.isfile(target_h):
            raise DistutilsSetupError('f2py target_h file %r not found' % (target_h,))
    for name_ext in ['-f2pywrappers.f', '-f2pywrappers2.f90']:
        filename = os.path.join(target_dir, ext_name + name_ext)
        if os.path.isfile(filename):
            log.info("  adding '%s' to sources." % filename)
            f_sources.append(filename)
    return new_sources + f_sources