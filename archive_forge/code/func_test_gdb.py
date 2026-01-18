import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb
def test_gdb():
    global have_gdb
    if have_gdb is not None:
        return have_gdb
    have_gdb = False
    try:
        p = subprocess.Popen(['gdb', '-nx', '--version'], stdout=subprocess.PIPE)
    except OSError:
        gdb_version = None
    else:
        stdout, _ = p.communicate()
        regex = 'GNU gdb [^\\d]*(\\d+)\\.(\\d+)'
        gdb_version = re.match(regex, stdout.decode('ascii', 'ignore'))
    if gdb_version:
        gdb_version_number = list(map(int, gdb_version.groups()))
        if gdb_version_number >= [7, 2]:
            have_gdb = True
            with tempfile.NamedTemporaryFile(mode='w+') as python_version_script:
                python_version_script.write('python import sys; print("%s %s" % sys.version_info[:2])')
                python_version_script.flush()
                p = subprocess.Popen(['gdb', '-batch', '-x', python_version_script.name], stdout=subprocess.PIPE)
                stdout, _ = p.communicate()
                try:
                    internal_python_version = list(map(int, stdout.decode('ascii', 'ignore').split()))
                    if internal_python_version < [2, 7]:
                        have_gdb = False
                except ValueError:
                    have_gdb = False
    if not have_gdb:
        warnings.warn('Skipping gdb tests, need gdb >= 7.2 with Python >= 2.7')
    return have_gdb