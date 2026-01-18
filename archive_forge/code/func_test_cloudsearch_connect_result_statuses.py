from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch.domain import Domain
from boto.cloudsearch.layer1 import Layer1
def test_cloudsearch_connect_result_statuses(self):
    """Check that domain statuses are correctly returned from AWS"""
    self.set_http_response(status_code=200)
    api_response = self.service_connection.create_domain('demo')
    domain = Domain(self, api_response)
    self.assertEqual(domain.created, True)
    self.assertEqual(domain.processing, False)
    self.assertEqual(domain.requires_index_documents, False)
    self.assertEqual(domain.deleted, False)