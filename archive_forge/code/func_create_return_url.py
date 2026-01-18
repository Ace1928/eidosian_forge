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
def create_return_url(base, query, **kwargs):
    """
    Add a query string plus extra parameters to a base URL which may contain
    a query part already.

    :param base: redirect_uri may contain a query part, no fragment allowed.
    :param query: Old query part as a string
    :param kwargs: extra query parameters
    :return:
    """
    part = urlsplit(base)
    if part.fragment:
        raise ValueError("Base URL contained parts it shouldn't")
    for key, values in parse_qs(query).items():
        if key in kwargs:
            if isinstance(kwargs[key], str):
                kwargs[key] = [kwargs[key]]
            kwargs[key].extend(values)
        else:
            kwargs[key] = values
    if part.query:
        for key, values in parse_qs(part.query).items():
            if key in kwargs:
                if isinstance(kwargs[key], str):
                    kwargs[key] = [kwargs[key]]
                kwargs[key].extend(values)
            else:
                kwargs[key] = values
        _pre = base.split('?')[0]
    else:
        _pre = base
    logger.debug(f'kwargs: {kwargs}')
    return f'{_pre}?{url_encode_params(kwargs)}'