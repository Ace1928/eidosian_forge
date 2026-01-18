from unittest import TestCase
import macaroonbakery.httpbakery as httpbakery
import requests
from mock import patch
from httmock import HTTMock, response, urlmatch
@patch('webbrowser.open')
def test_407_then_error_on_wait(self, mock_open):
    client = httpbakery.Client()
    with HTTMock(first_407_then_200), HTTMock(discharge_401), HTTMock(wait_on_error):
        with self.assertRaises(httpbakery.InteractionError) as exc:
            requests.get(ID_PATH, cookies=client.cookies, auth=client.auth())
    self.assertEqual(str(exc.exception), 'cannot start interactive session: cannot get http://example.com/wait')
    mock_open.assert_called_once_with(u'http://example.com/visit', new=1)