from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_create_domain(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    self.assert_request_parameters({'Action': 'CreateDomain', 'DomainName': 'demo', 'Version': '2011-02-01'})