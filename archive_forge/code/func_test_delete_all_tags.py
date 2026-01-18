import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_delete_all_tags(self):
    ref = self.new_ref()
    self.stub_url('PUT', parts=[self.collection_key, ref['id'], 'tags'], json={'tags': []}, status_code=200)
    ret = self.manager.update_tags(ref['id'], [])
    self.assertEqual([], ret)