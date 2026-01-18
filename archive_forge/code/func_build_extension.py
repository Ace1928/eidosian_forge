import os
import subprocess
from glob import glob
from distutils.dep_util import newer_group
from distutils.command.build_ext import build_ext as old_build_ext
from distutils.errors import DistutilsFileError, DistutilsSetupError,\
from distutils.file_util import copy_file
from numpy.distutils import log
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.system_info import combine_paths
from numpy.distutils.misc_util import (
from numpy.distutils.command.config_compiler import show_fortran_compilers
from numpy.distutils.ccompiler_opt import new_ccompiler_opt, CCompilerOpt
def build_extension(self, ext):
    sources = ext.sources
    if sources is None or not is_sequence(sources):
        raise DistutilsSetupError(("in 'ext_modules' option (extension '%s'), " + "'sources' must be present and must be " + 'a list of source filenames') % ext.name)
    sources = list(sources)
    if not sources:
        return
    fullname = self.get_ext_fullname(ext.name)
    if self.inplace:
        modpath = fullname.split('.')
        package = '.'.join(modpath[0:-1])
        base = modpath[-1]
        build_py = self.get_finalized_command('build_py')
        package_dir = build_py.get_package_dir(package)
        ext_filename = os.path.join(package_dir, self.get_ext_filename(base))
    else:
        ext_filename = os.path.join(self.build_lib, self.get_ext_filename(fullname))
    depends = sources + ext.depends
    force_rebuild = self.force
    if not self.disable_optimization and (not self.compiler_opt.is_cached()):
        log.debug('Detected changes on compiler optimizations')
        force_rebuild = True
    if not (force_rebuild or newer_group(depends, ext_filename, 'newer')):
        log.debug("skipping '%s' extension (up-to-date)", ext.name)
        return
    else:
        log.info("building '%s' extension", ext.name)
    extra_args = ext.extra_compile_args or []
    extra_cflags = getattr(ext, 'extra_c_compile_args', None) or []
    extra_cxxflags = getattr(ext, 'extra_cxx_compile_args', None) or []
    macros = ext.define_macros[:]
    for undef in ext.undef_macros:
        macros.append((undef,))
    c_sources, cxx_sources, f_sources, fmodule_sources = filter_sources(ext.sources)
    if self.compiler.compiler_type == 'msvc':
        if cxx_sources:
            extra_args.append('/Zm1000')
            extra_cflags += extra_cxxflags
        c_sources += cxx_sources
        cxx_sources = []
    if ext.language == 'f90':
        fcompiler = self._f90_compiler
    elif ext.language == 'f77':
        fcompiler = self._f77_compiler
    else:
        fcompiler = self._f90_compiler or self._f77_compiler
    if fcompiler is not None:
        fcompiler.extra_f77_compile_args = ext.extra_f77_compile_args or [] if hasattr(ext, 'extra_f77_compile_args') else []
        fcompiler.extra_f90_compile_args = ext.extra_f90_compile_args or [] if hasattr(ext, 'extra_f90_compile_args') else []
    cxx_compiler = self._cxx_compiler
    if cxx_sources and cxx_compiler is None:
        raise DistutilsError('extension %r has C++ sourcesbut no C++ compiler found' % ext.name)
    if (f_sources or fmodule_sources) and fcompiler is None:
        raise DistutilsError('extension %r has Fortran sources but no Fortran compiler found' % ext.name)
    if ext.language in ['f77', 'f90'] and fcompiler is None:
        self.warn('extension %r has Fortran libraries but no Fortran linker found, using default linker' % ext.name)
    if ext.language == 'c++' and cxx_compiler is None:
        self.warn('extension %r has C++ libraries but no C++ linker found, using default linker' % ext.name)
    kws = {'depends': ext.depends}
    output_dir = self.build_temp
    include_dirs = ext.include_dirs + get_numpy_include_dirs()
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
    c_objects = []
    if copt_cxx_sources:
        log.info('compiling C++ dispatch-able sources')
        c_objects += self.compiler_opt.try_dispatch(copt_cxx_sources, output_dir=output_dir, src_dir=copt_build_src, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_args + extra_cxxflags, ccompiler=cxx_compiler, **kws)
    if copt_c_sources:
        log.info('compiling C dispatch-able sources')
        c_objects += self.compiler_opt.try_dispatch(copt_c_sources, output_dir=output_dir, src_dir=copt_build_src, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_args + extra_cflags, **kws)
    if c_sources:
        log.info('compiling C sources')
        c_objects += self.compiler.compile(c_sources, output_dir=output_dir, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_args + copt_baseline_flags + extra_cflags, **kws)
    if cxx_sources:
        log.info('compiling C++ sources')
        c_objects += cxx_compiler.compile(cxx_sources, output_dir=output_dir, macros=macros + copt_macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_args + copt_baseline_flags + extra_cxxflags, **kws)
    extra_postargs = []
    f_objects = []
    if fmodule_sources:
        log.info('compiling Fortran 90 module sources')
        module_dirs = ext.module_dirs[:]
        module_build_dir = os.path.join(self.build_temp, os.path.dirname(self.get_ext_filename(fullname)))
        self.mkpath(module_build_dir)
        if fcompiler.module_dir_switch is None:
            existing_modules = glob('*.mod')
        extra_postargs += fcompiler.module_options(module_dirs, module_build_dir)
        f_objects += fcompiler.compile(fmodule_sources, output_dir=self.build_temp, macros=macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs, depends=ext.depends)
        if fcompiler.module_dir_switch is None:
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
        f_objects += fcompiler.compile(f_sources, output_dir=self.build_temp, macros=macros, include_dirs=include_dirs, debug=self.debug, extra_postargs=extra_postargs, depends=ext.depends)
    if f_objects and (not fcompiler.can_ccompiler_link(self.compiler)):
        unlinkable_fobjects = f_objects
        objects = c_objects
    else:
        unlinkable_fobjects = []
        objects = c_objects + f_objects
    if ext.extra_objects:
        objects.extend(ext.extra_objects)
    extra_args = ext.extra_link_args or []
    libraries = self.get_libraries(ext)[:]
    library_dirs = ext.library_dirs[:]
    linker = self.compiler.link_shared_object
    if self.compiler.compiler_type in ('msvc', 'intelw', 'intelemw'):
        self._libs_with_msvc_and_fortran(fcompiler, libraries, library_dirs)
        if ext.runtime_library_dirs:
            for d in ext.runtime_library_dirs:
                for f in glob(d + '/*.dll'):
                    copy_file(f, self.extra_dll_dir)
            ext.runtime_library_dirs = []
    elif ext.language in ['f77', 'f90'] and fcompiler is not None:
        linker = fcompiler.link_shared_object
    if ext.language == 'c++' and cxx_compiler is not None:
        linker = cxx_compiler.link_shared_object
    if fcompiler is not None:
        objects, libraries = self._process_unlinkable_fobjects(objects, libraries, fcompiler, library_dirs, unlinkable_fobjects)
    linker(objects, ext_filename, libraries=libraries, library_dirs=library_dirs, runtime_library_dirs=ext.runtime_library_dirs, extra_postargs=extra_args, export_symbols=self.get_export_symbols(ext), debug=self.debug, build_temp=self.build_temp, target_lang=ext.language)