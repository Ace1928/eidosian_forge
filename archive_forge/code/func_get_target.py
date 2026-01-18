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
def get_target(self):
    try:
        p = subprocess.Popen(self.compiler_f77 + ['-v'], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = p.communicate()
        output = (stdout or b'') + (stderr or b'')
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        output = filepath_from_subprocess_output(output)
        m = TARGET_R.search(output)
        if m:
            return m.group(1)
    return ''