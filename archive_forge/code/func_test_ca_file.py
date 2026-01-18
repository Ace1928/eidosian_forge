import threading
from base64 import b64encode
from datetime import datetime
from time import sleep
import certifi
import pytest
import responses
from kivy.network.urlrequest import UrlRequestRequests as UrlRequest
from requests.auth import HTTPBasicAuth
from responses import matchers
@pytest.mark.parametrize('scheme', ('http', 'https'))
@responses.activate
def test_ca_file(self, scheme, kivy_clock):
    _queue = UrlRequestQueue([])
    responses.get(f'{scheme}://example.com', body='{}', status=400, content_type='application/json')
    req = UrlRequest(f'{scheme}://example.com', on_finish=_queue._on_finish, ca_file=certifi.where(), debug=True)
    self.wait_request_is_finished(kivy_clock, req)
    processed_queue = _queue.queue
    assert len(processed_queue) == 1
    self._ensure_called_from_thread(processed_queue)
    self._check_queue_values(processed_queue[0], 'finish')