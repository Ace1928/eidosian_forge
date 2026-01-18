from http import client as http_client
from oslotest import base as test_base
from ironicclient.common.apiclient import exceptions
def test_from_response_unknown(self):
    method = 'POST'
    url = '/fake-unknown'
    status_code = 499
    json_data = {'error': {'message': 'fake unknown message', 'details': 'fake unknown details'}}
    self.assert_exception(exceptions.HTTPClientError, method, url, status_code, json_data)
    status_code = 600
    self.assert_exception(exceptions.HttpError, method, url, status_code, json_data)