import re
import os
import sys
import warnings
import platform
import tempfile
import hashlib
import base64
import subprocess
from subprocess import Popen, PIPE, STDOUT
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.fcompiler import FCompiler
from distutils.version import LooseVersion
def gnu_version_match(self, version_string):
    """Handle the different versions of GNU fortran compilers"""
    while version_string.startswith('gfortran: warning'):
        version_string = version_string[version_string.find('\n') + 1:].strip()
    if len(version_string) <= 20:
        m = re.search('([0-9.]+)', version_string)
        if m:
            if version_string.startswith('GNU Fortran'):
                return ('g77', m.group(1))
            elif m.start() == 0:
                return ('gfortran', m.group(1))
    else:
        m = re.search('GNU Fortran\\s+95.*?([0-9-.]+)', version_string)
        if m:
            return ('gfortran', m.group(1))
        m = re.search('GNU Fortran.*?\\-?([0-9-.]+\\.[0-9-.]+)', version_string)
        if m:
            v = m.group(1)
            if v.startswith('0') or v.startswith('2') or v.startswith('3'):
                return ('g77', v)
            else:
                return ('gfortran', v)
    err = 'A valid Fortran version was not found in this string:\n'
    raise ValueError(err + version_string)