import json
import mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudsearch2.domain import Domain
from boto.cloudsearch2.layer1 import CloudSearchConnection
from boto.cloudsearchdomain.layer1 import CloudSearchDomainConnection
def test_get_document_service(self):
    layer1 = CloudSearchConnection(aws_access_key_id='aws_access_key_id', aws_secret_access_key='aws_secret_access_key', sign_request=True)
    domain = Domain(layer1=layer1, data=json.loads(self.domain_status))
    document_service = domain.get_document_service()
    self.assertEqual(document_service.sign_request, True)