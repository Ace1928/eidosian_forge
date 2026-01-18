from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.urls import urlparse
from ansible.module_utils.urls import generic_urlparse
from ansible.module_utils.urls import Request
from .common import F5ModuleError
from ansible.module_utils.six.moves.urllib.error import HTTPError
from .constants import (
@property
def raw_content(self):
    return self._content