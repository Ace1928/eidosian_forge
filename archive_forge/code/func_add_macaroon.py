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
def add_macaroon(data):
    try:
        data = utils.b64decode(data)
        data_as_objs = json.loads(data.decode('utf-8'))
    except ValueError:
        return
    ms = [utils.macaroon_from_dict(x) for x in data_as_objs]
    mss.append(ms)