from __future__ import absolute_import
import types
from . import Errors
def calc_str(self):
    if self.nocase:
        name = 'NoCase'
    else:
        name = 'Case'
    return '%s(%s)' % (name, self.re)