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
def install_from_zip(zip_fname, install_path, pkg_finder=None, py_lib_sdir=PY_LIB_SDIR, script_sdir='bin'):
    """Install package from zip file `zip_fname`

    Parameters
    ----------
    zip_fname : str
        filename of zip file containing package code
    install_path : str
        output prefix at which to install package
    pkg_finder : None or callable, optional
        If None, assume zip contains ``setup.py`` at the top level.  Otherwise,
        find directory containing ``setup.py`` with ``pth =
        pkg_finder(unzip_path)`` where ``unzip_path`` is the path to which we
        have unzipped the zip file contents.
    py_lib_sdir : str, optional
        subdirectory to which to write the library code from the package.  Thus
        if package called ``nibabel``, the written code will be in
        ``<install_path>/<py_lib_sdir>/nibabel
    script_sdir : str, optional
        subdirectory to which we write the installed scripts.  Thus scripts will
        be written to ``<install_path>/<script_sdir>
    """
    unzip_path = tempfile.mkdtemp()
    try:
        zip_extract_all(zip_fname, unzip_path)
        if pkg_finder is None:
            from_path = unzip_path
        else:
            from_path = pkg_finder(unzip_path)
        install_from_to(from_path, install_path, py_lib_sdir, script_sdir)
    finally:
        shutil.rmtree(unzip_path)