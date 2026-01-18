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
def filter_files(self, sources, exts=[]):
    new_sources = []
    files = []
    for source in sources:
        base, ext = os.path.splitext(source)
        if ext in exts:
            files.append(source)
        else:
            new_sources.append(source)
    return (new_sources, files)