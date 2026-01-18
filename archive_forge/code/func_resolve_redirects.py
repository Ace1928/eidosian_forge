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
def resolve_redirects(self, resp, req, stream=False, timeout=None, verify=True, cert=None, proxies=None, yield_requests=False, **adapter_kwargs):
    """Receives a Response. Returns a generator of Responses or Requests."""
    hist = []
    url = self.get_redirect_target(resp)
    previous_fragment = urlparse(req.url).fragment
    while url:
        prepared_request = req.copy()
        hist.append(resp)
        resp.history = hist[1:]
        try:
            resp.content
        except (ChunkedEncodingError, ContentDecodingError, RuntimeError):
            resp.raw.read(decode_content=False)
        if len(resp.history) >= self.max_redirects:
            raise TooManyRedirects('Exceeded {} redirects.'.format(self.max_redirects), response=resp)
        resp.close()
        if url.startswith('//'):
            parsed_rurl = urlparse(resp.url)
            url = ':'.join([to_native_string(parsed_rurl.scheme), url])
        parsed = urlparse(url)
        if parsed.fragment == '' and previous_fragment:
            parsed = parsed._replace(fragment=previous_fragment)
        elif parsed.fragment:
            previous_fragment = parsed.fragment
        url = parsed.geturl()
        if not parsed.netloc:
            url = urljoin(resp.url, requote_uri(url))
        else:
            url = requote_uri(url)
        prepared_request.url = to_native_string(url)
        self.rebuild_method(prepared_request, resp)
        if resp.status_code not in (codes.temporary_redirect, codes.permanent_redirect):
            purged_headers = ('Content-Length', 'Content-Type', 'Transfer-Encoding')
            for header in purged_headers:
                prepared_request.headers.pop(header, None)
            prepared_request.body = None
        headers = prepared_request.headers
        headers.pop('Cookie', None)
        extract_cookies_to_jar(prepared_request._cookies, req, resp.raw)
        merge_cookies(prepared_request._cookies, self.cookies)
        prepared_request.prepare_cookies(prepared_request._cookies)
        proxies = self.rebuild_proxies(prepared_request, proxies)
        self.rebuild_auth(prepared_request, resp)
        rewindable = prepared_request._body_position is not None and ('Content-Length' in headers or 'Transfer-Encoding' in headers)
        if rewindable:
            rewind_body(prepared_request)
        req = prepared_request
        if yield_requests:
            yield req
        else:
            resp = self.send(req, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies, allow_redirects=False, **adapter_kwargs)
            extract_cookies_to_jar(self.cookies, prepared_request, resp.raw)
            url = self.get_redirect_target(resp)
            yield resp