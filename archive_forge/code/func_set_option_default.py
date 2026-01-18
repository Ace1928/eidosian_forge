from __future__ import absolute_import, division, print_function
import os
def set_option_default(self, key, default=None):
    return self._defaultsetter(key, default)