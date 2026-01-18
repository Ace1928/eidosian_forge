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
def build_library_sources(self, lib_name, build_info):
    sources = list(build_info.get('sources', []))
    if not sources:
        return
    log.info('building library "%s" sources' % lib_name)
    sources = self.generate_sources(sources, (lib_name, build_info))
    sources = self.template_sources(sources, (lib_name, build_info))
    sources, h_files = self.filter_h_files(sources)
    if h_files:
        log.info('%s - nothing done with h_files = %s', self.package, h_files)
    build_info['sources'] = sources
    return