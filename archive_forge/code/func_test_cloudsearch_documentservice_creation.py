from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_documentservice_creation(self):
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    domain = Domain(self, api_response)
    document = domain.get_document_service()
    self.assertEqual(document.endpoint, 'doc-demo-userdomain.us-east-1.cloudsearch.amazonaws.com')