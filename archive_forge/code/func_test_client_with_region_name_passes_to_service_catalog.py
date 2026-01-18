import uuid
from oslo_serialization import jsonutils
from keystoneauth1 import fixture
from keystoneauth1 import session as auth_session
from keystoneclient.auth import token_endpoint
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient.tests.unit.v2_0 import utils
from keystoneclient.v2_0 import client
def test_client_with_region_name_passes_to_service_catalog(self):
    self.stub_auth(json=client_fixtures.auth_response_body())
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL, region_name='North')
    self.assertEqual(cl.service_catalog.url_for(service_type='image'), 'https://image.north.host/v1/')
    with self.deprecations.expect_deprecations_here():
        cl = client.Client(username='exampleuser', password='password', project_name='exampleproject', auth_url=self.TEST_URL, region_name='South')
    self.assertEqual(cl.service_catalog.url_for(service_type='image'), 'https://image.south.host/v1/')