import sys
import os
import re
import logging
from .errors import DistutilsOptionError
from . import util, dir_util, file_util, archive_util, _modified
from ._log import log
def get_command_name(self):
    if hasattr(self, 'command_name'):
        return self.command_name
    else:
        return self.__class__.__name__