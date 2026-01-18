import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_subtree_as_ids(self):
    projects = self._create_projects_hierarchy()
    ref = projects[0]
    ref['subtree'] = {projects[1]['id']: {projects[2]['id']: None}}
    self.stub_entity('GET', id=ref['id'], entity=ref)
    returned = self.manager.get(ref['id'], subtree_as_ids=True)
    self.assertQueryStringIs('subtree_as_ids')
    self.assertEqual(ref['subtree'], returned.subtree)