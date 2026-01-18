import os
import re
import sys
import copy
import glob
import atexit
import tempfile
import subprocess
import shutil
import multiprocessing
import textwrap
import importlib.util
from threading import local as tlocal
from functools import reduce
import distutils
from distutils.errors import DistutilsError
def get_pkg_info(pkgname, dirs=None):
    """
    Return library info for the given package.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of additional directories where to look
        for npy-pkg-config files. Those directories are searched prior to the
        NumPy directory.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    Configuration.add_npy_pkg_config, Configuration.add_installed_library,
    get_info

    """
    from numpy.distutils.npy_pkg_config import read_config
    if dirs:
        dirs.append(get_npy_pkg_dir())
    else:
        dirs = [get_npy_pkg_dir()]
    return read_config(pkgname, dirs)