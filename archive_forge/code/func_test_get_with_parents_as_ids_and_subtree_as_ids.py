import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_parents_as_ids_and_subtree_as_ids(self):
    ref = self.new_ref()
    projects = self._create_projects_hierarchy()
    ref = projects[1]
    ref['parents'] = {projects[0]['id']: None}
    ref['subtree'] = {projects[2]['id']: None}
    self.stub_entity('GET', id=ref['id'], entity=ref)
    returned = self.manager.get(ref['id'], parents_as_ids=True, subtree_as_ids=True)
    self.assertQueryStringIs('subtree_as_ids&parents_as_ids')
    self.assertEqual(ref['parents'], returned.parents)
    self.assertEqual(ref['subtree'], returned.subtree)