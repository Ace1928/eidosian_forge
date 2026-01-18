import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_check_tag(self):
    ref = self.new_ref()
    tag_name = 'blue'
    self.stub_url('HEAD', parts=[self.collection_key, ref['id'], 'tags', tag_name], status_code=204)
    self.assertTrue(self.manager.check_tag(ref['id'], tag_name))
    no_tag = 'orange'
    self.stub_url('HEAD', parts=[self.collection_key, ref['id'], 'tags', no_tag], status_code=404)
    self.assertFalse(self.manager.check_tag(ref['id'], no_tag))