from __future__ import absolute_import
import types
from . import Errors
def check_char(self, num, value):
    self.check_string(num, value)
    if len(value) != 1:
        raise Errors.PlexValueError('Invalid value for argument %d of Plex.%s.Expected a string of length 1, got: %s' % (num, self.__class__.__name__, repr(value)))