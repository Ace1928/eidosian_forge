import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def authorize_redirect(self, redirect_uri: Optional[str]=None, client_id: Optional[str]=None, client_secret: Optional[str]=None, extra_params: Optional[Dict[str, Any]]=None, scope: Optional[List[str]]=None, response_type: str='code') -> None:
    """Redirects the user to obtain OAuth authorization for this service.

        Some providers require that you register a redirect URL with
        your application instead of passing one via this method. You
        should call this method to log the user in, and then call
        ``get_authenticated_user`` in the handler for your
        redirect URL to complete the authorization process.

        .. versionchanged:: 6.0

           The ``callback`` argument and returned awaitable were removed;
           this is now an ordinary synchronous function.

        .. deprecated:: 6.4
           The ``client_secret`` argument (which has never had any effect)
           is deprecated and will be removed in Tornado 7.0.
        """
    if client_secret is not None:
        warnings.warn('client_secret argument is deprecated', DeprecationWarning)
    handler = cast(RequestHandler, self)
    args = {'response_type': response_type}
    if redirect_uri is not None:
        args['redirect_uri'] = redirect_uri
    if client_id is not None:
        args['client_id'] = client_id
    if extra_params:
        args.update(extra_params)
    if scope:
        args['scope'] = ' '.join(scope)
    url = self._OAUTH_AUTHORIZE_URL
    handler.redirect(url_concat(url, args))