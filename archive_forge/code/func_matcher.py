import os
import re
import sys
import platform
import shlex
import time
import subprocess
from copy import copy
from pathlib import Path
from distutils import ccompiler
from distutils.ccompiler import (
from distutils.errors import (
from distutils.sysconfig import customize_compiler
from distutils.version import LooseVersion
from numpy.distutils import log
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import cyg2win32, is_sequence, mingw32, \
import threading
def matcher(version_string):
    m = re.match(pat, version_string)
    if not m:
        return None
    version = m.group('version')
    return version