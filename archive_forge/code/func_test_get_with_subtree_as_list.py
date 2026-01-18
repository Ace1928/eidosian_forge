import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_subtree_as_list(self):
    projects = self._create_projects_hierarchy()
    ref = projects[0]
    ref['subtree_as_list'] = []
    for i in range(1, len(projects)):
        ref['subtree_as_list'].append(projects[i])
    self.stub_entity('GET', id=ref['id'], entity=ref)
    returned = self.manager.get(ref['id'], subtree_as_list=True)
    self.assertQueryStringIs('subtree_as_list')
    for i in range(1, len(projects)):
        for attr in projects[i]:
            child = getattr(returned, 'subtree_as_list')[i - 1]
            self.assertEqual(child[attr], projects[i][attr], 'Expected different %s' % attr)