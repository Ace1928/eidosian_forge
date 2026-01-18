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
def compile_sources(files, Runner=None, destdir=None, cwd=None, keep_dir_struct=False, per_file_kwargs=None, **kwargs):
    """ Compile source code files to object files.

    Parameters
    ==========

    files : iterable of str
        Paths to source files, if ``cwd`` is given, the paths are taken as relative.
    Runner: CompilerRunner subclass (optional)
        Could be e.g. ``FortranCompilerRunner``. Will be inferred from filename
        extensions if missing.
    destdir: str
        Output directory, if cwd is given, the path is taken as relative.
    cwd: str
        Working directory. Specify to have compiler run in other directory.
        also used as root of relative paths.
    keep_dir_struct: bool
        Reproduce directory structure in `destdir`. default: ``False``
    per_file_kwargs: dict
        Dict mapping instances in ``files`` to keyword arguments.
    \\*\\*kwargs: dict
        Default keyword arguments to pass to ``Runner``.

    """
    _per_file_kwargs = {}
    if per_file_kwargs is not None:
        for k, v in per_file_kwargs.items():
            if isinstance(k, Glob):
                for path in glob.glob(k.pathname):
                    _per_file_kwargs[path] = v
            elif isinstance(k, ArbitraryDepthGlob):
                for path in glob_at_depth(k.filename, cwd):
                    _per_file_kwargs[path] = v
            else:
                _per_file_kwargs[k] = v
    destdir = destdir or '.'
    if not os.path.isdir(destdir):
        if os.path.exists(destdir):
            raise OSError('{} is not a directory'.format(destdir))
        else:
            make_dirs(destdir)
    if cwd is None:
        cwd = '.'
        for f in files:
            copy(f, destdir, only_update=True, dest_is_dir=True)
    dstpaths = []
    for f in files:
        if keep_dir_struct:
            name, ext = os.path.splitext(f)
        else:
            name, ext = os.path.splitext(os.path.basename(f))
        file_kwargs = kwargs.copy()
        file_kwargs.update(_per_file_kwargs.get(f, {}))
        dstpaths.append(src2obj(f, Runner, cwd=cwd, **file_kwargs))
    return dstpaths