import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def iter_info():
    make_bold = _make_bold_win32 if os.name == 'nt' else _make_bold_unix
    yield make_bold('rpy2 version:')
    try:
        import rpy2
        yield rpy2.__version__
    except ImportError:
        yield 'rpy2 cannot be imported'
    yield make_bold('Python version:')
    yield sys.version
    yield make_bold("Looking for R's HOME:")
    r_home = os.environ.get('R_HOME')
    yield ('    Environment variable R_HOME: %s' % r_home)
    r_home_default = None
    if os.name == 'nt':
        r_home_default = r_home_from_registry()
        yield ('    InstallPath in the registry: %s' % r_home_default)
        r_user = os.environ.get('R_USER')
        yield ('    Environment variable R_USER: %s' % r_user)
    else:
        try:
            r_home_default = r_home_from_subprocess()
        except Exception as e:
            logger.error(f'Unable to determine R home: {e}')
        yield ('    Calling `R RHOME`: %s' % r_home_default)
    yield ('    Environment variable R_LIBS_USER: %s' % os.environ.get('R_LIBS_USER'))
    if r_home is not None and r_home_default is not None:
        if os.path.abspath(r_home) != r_home_default:
            yield '    Warning: The environment variable R_HOME differs from the default R in the PATH.'
    elif r_home_default is None:
        yield '    Warning: There is no R in the PATH and no R_HOME defined.'
    else:
        r_home = r_home_default
    if os.name != 'nt':
        yield make_bold("R's value for LD_LIBRARY_PATH:")
        if r_home is None:
            yield '     *** undefined when not R home can be determined'
        else:
            yield r_ld_library_path_from_subprocess(r_home)
    try:
        import rpy2.rinterface_lib.openrlib
        rlib_status = 'OK'
    except ImportError as ie:
        try:
            import rpy2
            rlib_status = '*** Error while loading: %s ***' % str(ie)
        except ImportError:
            rlib_status = '*** rpy2 is not installed'
    except OSError as ose:
        rlib_status = str(ose)
    yield make_bold('R version:')
    yield ('    In the PATH: %s' % r_version_from_subprocess())
    yield ('    Loading R library from rpy2: %s' % rlib_status)
    r_libs = os.environ.get('R_LIBS')
    yield make_bold('Additional directories to load R packages from:')
    yield r_libs
    yield make_bold('C extension compilation:')
    c_ext = CExtensionOptions()
    if r_home is None:
        yield '    Warning: R cannot be found, so no compilation flags can be extracted.'
    else:
        try:
            c_ext.add_lib(*get_r_flags(r_home, '--ldflags'))
            c_ext.add_include(*get_r_flags(r_home, '--cppflags'))
            yield '  include:'
            yield ('  %s' % c_ext.include_dirs)
            yield '  libraries:'
            yield ('  %s' % c_ext.libraries)
            yield '  library_dirs:'
            yield ('  %s' % c_ext.library_dirs)
            yield '  extra_compile_args:'
            yield ('  %s' % c_ext.extra_compile_args)
            yield '  extra_link_args:'
            yield ('  %s' % c_ext.extra_link_args)
        except subprocess.CalledProcessError:
            yield '    Warning: Unable to get R compilation flags.'
    yield 'Directory for the R shared library:'
    yield get_r_libnn(r_home)
    yield make_bold('CFFI extension type')
    yield f'  Environment variable: {ENVVAR_CFFI_TYPE}'
    yield f'  Value: {get_cffi_mode()}'
    import importlib
    for cffi_type in ('abi', 'api'):
        rinterface_cffi_spec = importlib.util.find_spec(f'_rinterface_cffi_{cffi_type}')
        yield f'  {cffi_type.upper()}: {('PRESENT' if rinterface_cffi_spec else 'ABSENT')}'