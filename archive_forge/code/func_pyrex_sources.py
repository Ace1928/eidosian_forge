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
def pyrex_sources(self, sources, extension):
    """Pyrex not supported; this remains for Cython support (see below)"""
    new_sources = []
    ext_name = extension.name.split('.')[-1]
    for source in sources:
        base, ext = os.path.splitext(source)
        if ext == '.pyx':
            target_file = self.generate_a_pyrex_source(base, ext_name, source, extension)
            new_sources.append(target_file)
        else:
            new_sources.append(source)
    return new_sources