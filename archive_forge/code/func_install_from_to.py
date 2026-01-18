import os
import re
import shutil
import sys
import tempfile
import zipfile
from glob import glob
from os.path import abspath
from os.path import join as pjoin
from subprocess import PIPE, Popen
import os
import sys
import {mod_name}
def install_from_to(from_dir, to_dir, py_lib_sdir=PY_LIB_SDIR, bin_sdir='bin'):
    """Install package in `from_dir` to standard location in `to_dir`

    Parameters
    ----------
    from_dir : str
        path containing files to install with ``python setup.py ...``
    to_dir : str
        prefix path to which files will be installed, as in ``python setup.py
        install --prefix=to_dir``
    py_lib_sdir : str, optional
        subdirectory within `to_dir` to which library code will be installed
    bin_sdir : str, optional
        subdirectory within `to_dir` to which scripts will be installed
    """
    site_pkgs_path = os.path.join(to_dir, py_lib_sdir)
    py_lib_locs = f' --install-purelib={site_pkgs_path} --install-platlib={site_pkgs_path}'
    pwd = os.path.abspath(os.getcwd())
    cmd = f'{PYTHON} setup.py --quiet install --prefix={to_dir} {py_lib_locs}'
    try:
        os.chdir(from_dir)
        back_tick(cmd)
    finally:
        os.chdir(pwd)