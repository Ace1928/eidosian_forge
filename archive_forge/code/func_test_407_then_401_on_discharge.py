from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@patch('webbrowser.open')
def test_407_then_401_on_discharge(self, mock_open):
    client = httpbakery.Client()
    with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(wait_after_401):
        resp = requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
        resp.raise_for_status()
    mock_open.assert_called_once_with(u'http://example.com/visit', new=1)
    assert 'macaroon-test' in client.cookies.keys()