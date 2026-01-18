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
def compile_link_import_strings(sources, build_dir=None, **kwargs):
    """ Compiles, links and imports extension module from source.

    Parameters
    ==========

    sources : iterable of name/source pair tuples
    build_dir : string (default: None)
        Path. ``None`` implies use a temporary directory.
    **kwargs:
        Keyword arguments passed onto `compile_link_import_py_ext`.

    Returns
    =======

    mod : module
        The compiled and imported extension module.
    info : dict
        Containing ``build_dir`` as 'build_dir'.

    """
    source_files, build_dir = _write_sources_to_build_dir(sources, build_dir)
    mod = compile_link_import_py_ext(source_files, build_dir=build_dir, **kwargs)
    info = {'build_dir': build_dir}
    return (mod, info)