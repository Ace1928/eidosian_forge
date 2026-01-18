from http import client as http_client
from oslotest import base as test_base
from ironicclient.common.apiclient import exceptions
def test_from_response_with_different_response_format(self):
    method = 'GET'
    url = '/fake-wsme'
    status_code = http_client.BAD_REQUEST
    json_data1 = {'error_message': {'debuginfo': None, 'faultcode': 'Client', 'faultstring': 'fake message'}}
    message = str(json_data1['error_message']['faultstring'])
    details = str(json_data1)
    self.assert_exception(exceptions.BadRequest, method, url, status_code, json_data1, message, details)
    json_data2 = {'badRequest': {'message': 'fake message', 'code': http_client.BAD_REQUEST}}
    message = str(json_data2['badRequest']['message'])
    details = str(json_data2)
    self.assert_exception(exceptions.BadRequest, method, url, status_code, json_data2, message, details)