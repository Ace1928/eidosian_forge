from http import client as http_client
from oslotest import base as test_base
from ironicclient.common.apiclient import exceptions
def test_from_response_known(self):
    method = 'GET'
    url = '/fake'
    status_code = http_client.BAD_REQUEST
    json_data = {'error': {'message': 'fake message', 'details': 'fake details'}}
    self.assert_exception(exceptions.BadRequest, method, url, status_code, json_data)