import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def get_swig_modulename(source):
    with open(source) as f:
        name = None
        for line in f:
            m = _swig_module_name_match(line)
            if m:
                name = m.group('name')
                break
    return name