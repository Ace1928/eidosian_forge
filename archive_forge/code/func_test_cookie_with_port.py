from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
def test_cookie_with_port(self):
    client = httpbakery.Client()
    with HTTMock(first_407_then_200_with_port):
        with HTTMock(discharge_200):
            resp = requests.get('http://example.com:8000/someprotecteurl', cookies=client.cookies, auth=client.auth())
    resp.raise_for_status()
    assert 'macaroon-test' in client.cookies.keys()