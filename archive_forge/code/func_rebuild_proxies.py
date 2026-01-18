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
def rebuild_proxies(self, prepared_request, proxies):
    """This method re-evaluates the proxy configuration by considering the
        environment variables. If we are redirected to a URL covered by
        NO_PROXY, we strip the proxy configuration. Otherwise, we set missing
        proxy keys for this URL (in case they were stripped by a previous
        redirect).

        This method also replaces the Proxy-Authorization header where
        necessary.

        :rtype: dict
        """
    headers = prepared_request.headers
    scheme = urlparse(prepared_request.url).scheme
    new_proxies = resolve_proxies(prepared_request, proxies, self.trust_env)
    if 'Proxy-Authorization' in headers:
        del headers['Proxy-Authorization']
    try:
        username, password = get_auth_from_url(new_proxies[scheme])
    except KeyError:
        username, password = (None, None)
    if username and password:
        headers['Proxy-Authorization'] = _basic_auth_str(username, password)
    return new_proxies