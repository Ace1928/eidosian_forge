from __future__ import absolute_import, division, print_function
from . import errors, http
@property
def auth_header(self):
    if not self._auth_header:
        self._auth_header = self._login()
    return self._auth_header