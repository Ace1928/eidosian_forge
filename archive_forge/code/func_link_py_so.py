import glob
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from sysconfig import get_config_var, get_config_vars, get_path
from .runners import (
from .util import (
def link_py_so(obj_files, so_file=None, cwd=None, libraries=None, cplus=False, fort=False, **kwargs):
    """ Link Python extension module (shared object) for importing

    Parameters
    ==========

    obj_files: iterable of str
        Paths to object files to be linked.
    so_file: str
        Name (path) of shared object file to create. If not specified it will
        have the basname of the last object file in `obj_files` but with the
        extension '.so' (Unix).
    cwd: path string
        Root of relative paths and working directory of linker.
    libraries: iterable of strings
        Libraries to link against, e.g. ['m'].
    cplus: bool
        Any C++ objects? default: ``False``.
    fort: bool
        Any Fortran objects? default: ``False``.
    kwargs**: dict
        Keyword arguments passed to ``link(...)``.

    Returns
    =======

    Absolute path to the generate shared object.
    """
    libraries = libraries or []
    include_dirs = kwargs.pop('include_dirs', [])
    library_dirs = kwargs.pop('library_dirs', [])
    if sys.platform == 'win32':
        warnings.warn('Windows not yet supported.')
    elif sys.platform == 'darwin':
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]
        is_framework = False
        for opt in cfgDict['LIBS'].split():
            if is_framework:
                kwargs['linkline'] = kwargs.get('linkline', []) + ['-framework', opt]
                is_framework = False
            elif opt.startswith('-l'):
                libraries.append(opt[2:])
            elif opt.startswith('-framework'):
                is_framework = True
        libfile = cfgDict['LIBRARY']
        libname = '.'.join(libfile.split('.')[:-1])[3:]
        libraries.append(libname)
    elif sys.platform[:3] == 'aix':
        pass
    elif get_config_var('Py_ENABLE_SHARED'):
        cfgDict = get_config_vars()
        kwargs['linkline'] = kwargs.get('linkline', []) + [cfgDict['LDFLAGS']]
        library_dirs += [cfgDict['LIBDIR']]
        for opt in cfgDict['BLDLIBRARY'].split():
            if opt.startswith('-l'):
                libraries += [opt[2:]]
    else:
        pass
    flags = kwargs.pop('flags', [])
    needed_flags = ('-pthread',)
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    return link(obj_files, shared=True, flags=flags, cwd=cwd, cplus=cplus, fort=fort, include_dirs=include_dirs, libraries=libraries, library_dirs=library_dirs, **kwargs)