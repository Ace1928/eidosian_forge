from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
def assert_cookie_security(self, cookies, name, secure):
    for cookie in cookies:
        if cookie.name == name:
            assert cookie.secure == secure
            break
    else:
        assert False, 'no cookie named {} found in jar'.format(name)