from requests import auth
from requests import cookies
from . import _digest_auth_compat as auth_compat, http_proxy_digest
def handle_401(self, r, **kwargs):
    """Resends a request with auth headers, if needed."""
    www_authenticate = r.headers.get('www-authenticate', '').lower()
    if 'basic' in www_authenticate:
        return self._handle_basic_auth_401(r, kwargs)
    if 'digest' in www_authenticate:
        return self._handle_digest_auth_401(r, kwargs)