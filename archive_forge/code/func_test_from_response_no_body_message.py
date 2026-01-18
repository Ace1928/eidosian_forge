import datetime
from unittest import mock
import requests
from cinderclient import exceptions
from cinderclient.tests.unit import utils
def test_from_response_no_body_message(self):
    response = requests.Response()
    response.status_code = 500
    body = {'keys': {}}
    ex = exceptions.from_response(response, body)
    self.assertIs(exceptions.ClientException, type(ex))
    self.assertEqual('n/a', ex.message)