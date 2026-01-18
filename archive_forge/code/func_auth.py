import base64
import json
import logging
import macaroonbakery.bakery as bakery
import macaroonbakery.checkers as checkers
import macaroonbakery._utils as utils
from ._browser import WebBrowserInteractor
from ._error import (
from ._interactor import (
import requests
from six.moves.http_cookies import SimpleCookie
from six.moves.urllib.parse import urljoin
def auth(self):
    """Return an authorizer object suitable for passing
        to requests methods that accept one.
        If a request returns a discharge-required error,
        the authorizer will acquire discharge macaroons
        and retry the request.
        """
    return _BakeryAuth(self)