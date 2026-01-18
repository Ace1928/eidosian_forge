import io
import logging
import os
from shlex import split as shsplit
import sys
import numpy
def make_extension(python, **extra):
    cfg = init_cfg('pythran.cfg', 'pythran-{}.cfg'.format(sys.platform), '.pythranrc', extra.get('config', None))
    if 'config' in extra:
        extra.pop('config')

    def parse_define(define):
        index = define.find('=')
        if index < 0:
            return (define, None)
        else:
            return (define[:index], define[index + 1:])
    extension = {'language': 'c++', 'define_macros': [str(x) for x in shsplit(cfg.get('compiler', 'defines'))], 'undef_macros': [str(x) for x in shsplit(cfg.get('compiler', 'undefs'))], 'include_dirs': [str(x) for x in shsplit(cfg.get('compiler', 'include_dirs'))], 'library_dirs': [str(x) for x in shsplit(cfg.get('compiler', 'library_dirs'))], 'libraries': [str(x) for x in shsplit(cfg.get('compiler', 'libs'))], 'extra_compile_args': [str(x) for x in shsplit(cfg.get('compiler', 'cflags'))], 'extra_link_args': [str(x) for x in shsplit(cfg.get('compiler', 'ldflags'))], 'extra_objects': []}
    if python:
        extension['define_macros'].append('ENABLE_PYTHON_MODULE')
    extension['define_macros'].append('__PYTHRAN__={}'.format(sys.version_info.major))
    pythonic_dir = get_include()
    extension['include_dirs'].append(pythonic_dir)
    extra.pop('language', None)
    cxx = extra.pop('cxx', None)
    cc = extra.pop('cc', None)
    if cxx is None:
        cxx = compiler()
    if cxx is not None:
        extension['cxx'] = cxx
        extension['cc'] = cc or cxx
    cflags = os.environ.get('CXXFLAGS', None)
    if cflags is not None:
        extension['extra_compile_args'].extend(shsplit(cflags))
    ldflags = os.environ.get('LDFLAGS', None)
    if ldflags is not None:
        extension['extra_link_args'].extend(shsplit(ldflags))
    for k, w in extra.items():
        extension[k].extend(w)
    if cfg.getboolean('pythran', 'complex_hook'):
        extension['include_dirs'].append(pythonic_dir + '/pythonic/patch')
    if python:
        extension['include_dirs'].append(numpy.get_include())
    reserved_blas_entries = ('pythran-openblas', 'none')
    user_blas = cfg.get('compiler', 'blas')
    if user_blas == 'pythran-openblas':
        try:
            import pythran_openblas as openblas
            extension['define_macros'].append('PYTHRAN_BLAS_OPENBLAS')
            extension['include_dirs'].extend(openblas.include_dirs)
            extension['extra_objects'].append(os.path.join(openblas.library_dir, openblas.static_library))
        except ImportError:
            logger.warning("Failed to find 'pythran-openblas' package. Please install it or change the compiler.blas setting. Defaulting to 'none'")
            user_blas = 'none'
    if user_blas == 'none':
        extension['define_macros'].append('PYTHRAN_BLAS_NONE')
    if user_blas not in reserved_blas_entries:
        try:
            import numpy.distutils.system_info as numpy_sys
            with silent():
                numpy_blas = numpy_sys.get_info(user_blas)
                extension['libraries'].extend(numpy_blas.get('libraries', []))
                extension['library_dirs'].extend(numpy_blas.get('library_dirs', []))
        except ImportError:
            blas = numpy.show_config('dicts')['Build Dependencies']['blas']
            libblas = {'openblas64': 'openblas'}.get(blas['name'], blas['name'])
            extension['libraries'].append(libblas)
    extension['define_macros'] = [dm if isinstance(dm, tuple) else parse_define(dm) for dm in extension['define_macros']]
    return extension