from __future__ import absolute_import
import types
from . import Errors
def check_re(self, num, value):
    if not isinstance(value, RE):
        self.wrong_type(num, value, 'Plex.RE instance')