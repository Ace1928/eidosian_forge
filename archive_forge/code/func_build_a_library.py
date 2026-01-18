import os
from glob import glob
import shutil
from distutils.command.build_clib import build_clib as old_build_clib
from distutils.errors import DistutilsSetupError, DistutilsError, \
from numpy.distutils import log
from distutils.dep_util import newer_group
from numpy.distutils.misc_util import (
from numpy.distutils.ccompiler_opt import new_ccompiler_opt
def build_a_library(self, build_info, lib_name, libraries):
    compiler = self.compiler
    fcompiler = self._f_compiler
    sources = build_info.get('sources')
    if sources is None or not is_sequence(sources):
        raise DistutilsSetupError(("in 'libraries' option (library '%s'), " + "'sources' must be present and must be " + 'a list of source filenames') % lib_name)
    sources = list(sources)
    c_sources, cxx_sources, f_sources, fmodule_sources = filter_sources(sources)
    requiref90 = not not fmodule_sources or build_info.get('language', 'c') == 'f90'
    source_languages = []
    if c_sources:
        source_languages.append('c')
    if cxx_sources:
        source_languages.append('c++')
    if requiref90:
        source_languages.append('f90')
    elif f_sources:
        source_languages.append('f77')
    build_info['source_languages'] = source_languages
    lib_file = compiler.library_filename(lib_name, output_dir=self.build_clib)
    depends = sources + build_info.get('depends', [])
    force_rebuild = self.force
    if not self.disable_optimization and (not self.compiler_opt.is_cached()):
        log.debug('Detected changes on compiler optimizations')
        force_rebuild = True
    if not (force_rebuild or newer_group(depends, lib_file, 'newer')):
        log.debug("skipping '%s' library (up-to-date)", lib_name)
        return
    else:
        log.info("building '%s' library", lib_name)
    config_fc = build_info.get('config_fc', {})
    if fcompiler is not None and config_fc:
        log.info('using additional config_fc from setup script for fortran compiler: %s' % (config_fc,))
        from numpy.distutils.fcompiler import new_fcompiler
        fcompiler = new_fcompiler(compiler=fcompiler.compiler_type, verbose=self.verbose, dry_run=self.dry_run, force=self.force, requiref90=requiref90, c_compiler=self.compiler)
        if fcompiler is not None:
            dist = self.distribution
            base_config_fc = dist.get_option_dict('config_fc').copy()
            base_config_fc.update(config_fc)
            fcompiler.customize(base_config_fc)
    if (f_sources or fmodule_sources) and fcompiler is None:
        raise DistutilsError('library %s has Fortran sources but no Fortran compiler found' % lib_name)
    if fcompiler is not None:
        fcompiler.extra_f77_compile_args = build_info.get('extra_f77_compile_args') or []
        fcompiler.extra_f90_compile_args = build_info.get('extra_f90_compile_args') or []
    macros = build_info.get('macros')
    if macros is None:
        macros = []
    include_dirs = build_info.get('include_dirs')
    if include_dirs is None:
        include_dirs = []
    extra_postargs = self.assemble_flags(build_info.get('extra_compiler_args'))
    extra_cflags = self.assemble_flags(build_info.get('extra_cflags'))
    extra_cxxflags = self.assemble_flags(build_info.get('extra_cxxflags'))
    include_dirs.extend(get_numpy_include_dirs())
    module_dirs = build_info.get('module_dirs') or []
    module_build_dir = os.path.dirname(lib_file)
    if requiref90:
        self.mkpath(module_build_dir)
    if compiler.compiler_type == 'msvc':
        c_sources += cxx_sources
        cxx_sources = []
        extra_cflags += extra_cxxflags
    copt_c_sources = []
    copt_cxx_sources = []
    copt_baseline_flags = []
    copt_macros = []
    if not self.disable_optimization:
        bsrc_dir = self.get_finalized_command('build_src').build_src
        dispatch_hpath = os.path.join('numpy', 'distutils', 'include')
        dispatch_hpath = os.path.join(bsrc_dir, dispatch_hpath)
        include_dirs.append(dispatch_hpath)
        copt_build_src = bsrc_dir
        for _srcs, _dst, _ext in (((c_sources,), copt_c_sources, ('.dispatch.c',)), ((c_sources, cxx_sources), copt_cxx_sources, ('.dispatch.cpp', '.dispatch.cxx'))):
            for _src in _srcs:
                _dst += [_src.pop(_src.index(s)) for s in _src[:] if s.endswith(_ext)]
        copt_baseline_flags = self.compiler_opt.cpu_baseline_flags()
    else:
        copt_macros.append(('NPY_DISABLE_OPTIMIZATION', 1))
    objects = []
    if copt_cxx_sources:
        log.info('compiling C++ dispatch-able sources')
        objects += self.compiler_opt.try_dispatch(copt_c_sources, output_dir=self.build_temp, src_dir=copt_build_src, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs + extra_cxxflags, ccompiler=cxx_compiler)
    if copt_c_sources:
        log.info('compiling C dispatch-able sources')
        objects += self.compiler_opt.try_dispatch(copt_c_sources, output_dir=self.build_temp, src_dir=copt_build_src, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs + extra_cflags)
    if c_sources:
        log.info('compiling C sources')
        objects += compiler.compile(c_sources, output_dir=self.build_temp, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs + copt_baseline_flags + extra_cflags)
    if cxx_sources:
        log.info('compiling C++ sources')
        cxx_compiler = compiler.cxx_compiler()
        cxx_objects = cxx_compiler.compile(cxx_sources, output_dir=self.build_temp, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs + copt_baseline_flags + extra_cxxflags)
        objects.extend(cxx_objects)
    if f_sources or fmodule_sources:
        extra_postargs = []
        f_objects = []
        if requiref90:
            if fcompiler.module_dir_switch is None:
                existing_modules = glob('*.mod')
            extra_postargs += fcompiler.module_options(module_dirs, module_build_dir)
        if fmodule_sources:
            log.info('compiling Fortran 90 module sources')
            f_objects += fcompiler.compile(fmodule_sources, output_dir=self.build_temp, macros=macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs)
        if requiref90 and self._f_compiler.module_dir_switch is None:
            for f in glob('*.mod'):
                if f in existing_modules:
                    continue
                t = os.path.join(module_build_dir, f)
                if os.path.abspath(f) == os.path.abspath(t):
                    continue
                if os.path.isfile(t):
                    os.remove(t)
                try:
                    self.move_file(f, module_build_dir)
                except DistutilsFileError:
                    log.warn('failed to move %r to %r' % (f, module_build_dir))
        if f_sources:
            log.info('compiling Fortran sources')
            f_objects += fcompiler.compile(f_sources, output_dir=self.build_temp, macros=macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs)
    else:
        f_objects = []
    if f_objects and (not fcompiler.can_ccompiler_link(compiler)):
        listfn = os.path.join(self.build_clib, lib_name + '.fobjects')
        with open(listfn, 'w') as f:
            f.write('\n'.join((os.path.abspath(obj) for obj in f_objects)))
        listfn = os.path.join(self.build_clib, lib_name + '.cobjects')
        with open(listfn, 'w') as f:
            f.write('\n'.join((os.path.abspath(obj) for obj in objects)))
        lib_fname = os.path.join(self.build_clib, lib_name + compiler.static_lib_extension)
        with open(lib_fname, 'wb') as f:
            pass
    else:
        objects.extend(f_objects)
        compiler.create_static_lib(objects, lib_name, output_dir=self.build_clib, debug=self.debug)
    clib_libraries = build_info.get('libraries', [])
    for lname, binfo in libraries:
        if lname in clib_libraries:
            clib_libraries.extend(binfo.get('libraries', []))
    if clib_libraries:
        build_info['libraries'] = clib_libraries