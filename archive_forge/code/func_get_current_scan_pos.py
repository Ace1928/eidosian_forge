from __future__ import absolute_import
import cython
from . import Errors
from .Regexps import BOL, EOL, EOF
def get_current_scan_pos(self):
    return self.current_scanner_position_tuple