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
def build_data_files_sources(self):
    if not self.data_files:
        return
    log.info('building data_files sources')
    from numpy.distutils.misc_util import get_data_files
    new_data_files = []
    for data in self.data_files:
        if isinstance(data, str):
            new_data_files.append(data)
        elif isinstance(data, tuple):
            d, files = data
            if self.inplace:
                build_dir = self.get_package_dir('.'.join(d.split(os.sep)))
            else:
                build_dir = os.path.join(self.build_src, d)
            funcs = [f for f in files if hasattr(f, '__call__')]
            files = [f for f in files if not hasattr(f, '__call__')]
            for f in funcs:
                if f.__code__.co_argcount == 1:
                    s = f(build_dir)
                else:
                    s = f()
                if s is not None:
                    if isinstance(s, list):
                        files.extend(s)
                    elif isinstance(s, str):
                        files.append(s)
                    else:
                        raise TypeError(repr(s))
            filenames = get_data_files((d, files))
            new_data_files.append((d, filenames))
        else:
            raise TypeError(repr(data))
    self.data_files[:] = new_data_files