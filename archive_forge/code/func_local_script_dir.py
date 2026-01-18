import os
import sys
from os.path import dirname, isdir, isfile
from os.path import join as pjoin
from os.path import pathsep, realpath
from subprocess import PIPE, Popen
def local_script_dir(script_sdir):
    """Get local script directory if running in development dir, else None"""
    package_path = dirname(__import__(MY_PACKAGE).__file__)
    above_us = realpath(pjoin(package_path, '..'))
    devel_script_dir = pjoin(above_us, script_sdir)
    if isfile(pjoin(above_us, 'setup.py')) and isdir(devel_script_dir):
        return devel_script_dir
    return None