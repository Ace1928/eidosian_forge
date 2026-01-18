import logging
import time
from urllib.parse import parse_qs
from urllib.parse import urlencode
from urllib.parse import urlsplit
from saml2 import SAMLError
import saml2.cryptography.symmetric
from saml2.httputil import Redirect
from saml2.httputil import Response
from saml2.httputil import Unauthorized
from saml2.httputil import make_cookie
from saml2.httputil import parse_cookie
def authenticated_as(self, cookie=None, **kwargs):
    if cookie is None:
        return None
    else:
        logger.debug(f'kwargs: {kwargs}')
        try:
            info, timestamp = parse_cookie(self.cookie_name, self.srv.seed, cookie)
            if self.active[info] == timestamp:
                msg = self.symmetric.decrypt(info).decode()
                uid, _ts = msg.split('::')
                if timestamp == _ts:
                    return {'uid': uid}
        except Exception:
            pass
    return None