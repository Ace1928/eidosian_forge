import cgi
import hashlib
import hmac
from http.cookies import SimpleCookie
import logging
import time
from typing import Optional
from urllib.parse import parse_qs
from urllib.parse import quote
from saml2 import BINDING_HTTP_ARTIFACT
from saml2 import BINDING_HTTP_POST
from saml2 import BINDING_HTTP_REDIRECT
from saml2 import BINDING_SOAP
from saml2 import BINDING_URI
from saml2 import SAMLError
from saml2 import time_util
def cookie_signature(seed, *parts):
    """Generates a cookie signature."""
    sha1 = hmac.new(seed, digestmod=hashlib.sha1)
    for part in parts:
        if part:
            sha1.update(part)
    return sha1.hexdigest()