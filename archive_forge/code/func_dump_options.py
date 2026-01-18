import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def dump_options(self, header=None, indent=''):
    from distutils.fancy_getopt import longopt_xlate
    if header is None:
        header = "command options for '%s':" % self.get_command_name()
    self.announce(indent + header, level=logging.INFO)
    indent = indent + '  '
    for option, _, _ in self.user_options:
        option = option.translate(longopt_xlate)
        if option[-1] == '=':
            option = option[:-1]
        value = getattr(self, option)
        self.announce(indent + '{} = {}'.format(option, value), level=logging.INFO)