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
def merge_hooks(request_hooks, session_hooks, dict_class=OrderedDict):
    """Properly merges both requests and session hooks.

    This is necessary because when request_hooks == {'response': []}, the
    merge breaks Session hooks entirely.
    """
    if session_hooks is None or session_hooks.get('response') == []:
        return request_hooks
    if request_hooks is None or request_hooks.get('response') == []:
        return session_hooks
    return merge_setting(request_hooks, session_hooks, dict_class)