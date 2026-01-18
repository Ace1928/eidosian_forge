import os
from configparser import ConfigParser
from distutils import log
from distutils.command.build_py import build_py
from distutils.command.install_scripts import install_scripts
from distutils.version import LooseVersion
from os.path import join as pjoin
from os.path import split as psplit
from os.path import splitext
def read_vars_from(ver_file):
    """Read variables from Python text file

    Parameters
    ----------
    ver_file : str
        Filename of file to read

    Returns
    -------
    info_vars : Bunch instance
        Bunch object where variables read from `ver_file` appear as
        attributes
    """
    ns = {}
    with open(ver_file, 'rt') as fobj:
        exec(fobj.read(), ns)
    return Bunch(ns)