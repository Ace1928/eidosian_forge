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
def pyx2obj(pyxpath, objpath=None, destdir=None, cwd=None, include_dirs=None, cy_kwargs=None, cplus=None, **kwargs):
    """
    Convenience function

    If cwd is specified, pyxpath and dst are taken to be relative
    If only_update is set to `True` the modification time is checked
    and compilation is only run if the source is newer than the
    destination

    Parameters
    ==========

    pyxpath: str
        Path to Cython source file.
    objpath: str (optional)
        Path to object file to generate.
    destdir: str (optional)
        Directory to put generated C file. When ``None``: directory of ``objpath``.
    cwd: str (optional)
        Working directory and root of relative paths.
    include_dirs: iterable of path strings (optional)
        Passed onto src2obj and via cy_kwargs['include_path']
        to simple_cythonize.
    cy_kwargs: dict (optional)
        Keyword arguments passed onto `simple_cythonize`
    cplus: bool (optional)
        Indicate whether C++ is used. default: auto-detect using ``.util.pyx_is_cplus``.
    compile_kwargs: dict
        keyword arguments passed onto src2obj

    Returns
    =======

    Absolute path of generated object file.

    """
    assert pyxpath.endswith('.pyx')
    cwd = cwd or '.'
    objpath = objpath or '.'
    destdir = destdir or os.path.dirname(objpath)
    abs_objpath = get_abspath(objpath, cwd=cwd)
    if os.path.isdir(abs_objpath):
        pyx_fname = os.path.basename(pyxpath)
        name, ext = os.path.splitext(pyx_fname)
        objpath = os.path.join(objpath, name + objext)
    cy_kwargs = cy_kwargs or {}
    cy_kwargs['output_dir'] = cwd
    if cplus is None:
        cplus = pyx_is_cplus(pyxpath)
    cy_kwargs['cplus'] = cplus
    interm_c_file = simple_cythonize(pyxpath, destdir=destdir, cwd=cwd, **cy_kwargs)
    include_dirs = include_dirs or []
    flags = kwargs.pop('flags', [])
    needed_flags = ('-fwrapv', '-pthread', '-fPIC')
    for flag in needed_flags:
        if flag not in flags:
            flags.append(flag)
    options = kwargs.pop('options', [])
    if kwargs.pop('strict_aliasing', False):
        raise CompileError('Cython requires strict aliasing to be disabled.')
    if cplus:
        std = kwargs.pop('std', 'c++98')
    else:
        std = kwargs.pop('std', 'c99')
    return src2obj(interm_c_file, objpath=objpath, cwd=cwd, include_dirs=include_dirs, flags=flags, std=std, options=options, inc_py=True, strict_aliasing=False, **kwargs)