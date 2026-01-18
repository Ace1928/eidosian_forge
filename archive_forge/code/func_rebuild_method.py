import os
import sys
import time
from datetime import timedelta
from collections import OrderedDict
from .auth import _basic_auth_str
from .compat import cookielib, is_py3, urljoin, urlparse, Mapping
from .cookies import (
from .models import Request, PreparedRequest, DEFAULT_REDIRECT_LIMIT
from .hooks import default_hooks, dispatch_hook
from ._internal_utils import to_native_string
from .utils import to_key_val_list, default_headers, DEFAULT_PORTS
from .exceptions import (
from .structures import CaseInsensitiveDict
from .adapters import HTTPAdapter
from .utils import (
from .status_codes import codes
from .models import REDIRECT_STATI
def rebuild_method(self, prepared_request, response):
    """When being redirected we may want to change the method of the request
        based on certain specs or browser behavior.
        """
    method = prepared_request.method
    if response.status_code == codes.see_other and method != 'HEAD':
        method = 'GET'
    if response.status_code == codes.found and method != 'HEAD':
        method = 'GET'
    if response.status_code == codes.moved and method == 'POST':
        method = 'GET'
    prepared_request.method = method