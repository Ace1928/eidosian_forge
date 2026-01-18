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
def template_sources(self, sources, extension):
    new_sources = []
    if is_sequence(extension):
        depends = extension[1].get('depends')
        include_dirs = extension[1].get('include_dirs')
    else:
        depends = extension.depends
        include_dirs = extension.include_dirs
    for source in sources:
        base, ext = os.path.splitext(source)
        if ext == '.src':
            if self.inplace:
                target_dir = os.path.dirname(base)
            else:
                target_dir = appendpath(self.build_src, os.path.dirname(base))
            self.mkpath(target_dir)
            target_file = os.path.join(target_dir, os.path.basename(base))
            if self.force or newer_group([source] + depends, target_file):
                if _f_pyf_ext_match(base):
                    log.info('from_template:> %s' % target_file)
                    outstr = process_f_file(source)
                else:
                    log.info('conv_template:> %s' % target_file)
                    outstr = process_c_file(source)
                with open(target_file, 'w') as fid:
                    fid.write(outstr)
            if _header_ext_match(target_file):
                d = os.path.dirname(target_file)
                if d not in include_dirs:
                    log.info("  adding '%s' to include_dirs." % d)
                    include_dirs.append(d)
            new_sources.append(target_file)
        else:
            new_sources.append(source)
    return new_sources