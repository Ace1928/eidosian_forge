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
def compile_run_strings(sources, build_dir=None, clean=False, compile_kwargs=None, link_kwargs=None):
    """ Compiles, links and runs a program built from sources.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    clean : bool
        Whether to remove build_dir after use. This will only have an
        effect if ``build_dir`` is ``None`` (which creates a temporary directory).
        Passing ``clean == True`` and ``build_dir != None`` raises a ``ValueError``.
        This will also set ``build_dir`` in returned info dictionary to ``None``.
    compile_kwargs: dict
        Keyword arguments passed onto ``compile_sources``
    link_kwargs: dict
        Keyword arguments passed onto ``link``

    Returns
    =======

    (stdout, stderr): pair of strings
    info: dict
        Containing exit status as 'exit_status' and ``build_dir`` as 'build_dir'

    """
    if clean and build_dir is not None:
        raise ValueError('Automatic removal of build_dir is only available for temporary directory.')
    try:
        source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
        objs = compile_sources(list(map(get_abspath, source_files)), destdir=build_dir, cwd=build_dir, **compile_kwargs or {})
        prog = link(objs, cwd=build_dir, fort=any_fortran_src(source_files), cplus=any_cplus_src(source_files), **link_kwargs or {})
        p = subprocess.Popen([prog], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        exit_status = p.wait()
        stdout, stderr = [txt.decode('utf-8') for txt in p.communicate()]
    finally:
        if clean and os.path.isdir(build_dir):
            shutil.rmtree(build_dir)
            build_dir = None
    info = {'exit_status': exit_status, 'build_dir': build_dir}
    return ((stdout, stderr), info)