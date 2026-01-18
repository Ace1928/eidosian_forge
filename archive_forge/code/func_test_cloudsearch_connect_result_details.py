from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_connect_result_details(self):
    """Check that the domain information is correctly returned from AWS"""
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    domain = Domain(self, api_response)
    self.assertEqual(domain.id, '1234567890/demo')
    self.assertEqual(domain.name, 'demo')