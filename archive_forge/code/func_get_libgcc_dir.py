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
def get_libgcc_dir(self):
    try:
        output = subprocess.check_output(self.compiler_f77 + ['-print-libgcc-file-name'])
    except (OSError, subprocess.CalledProcessError):
        pass
    else:
        output = filepath_from_subprocess_output(output)
        return os.path.dirname(output)
    return None