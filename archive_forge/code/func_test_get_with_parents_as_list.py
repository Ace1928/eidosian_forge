import fixtures
import uuid
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import projects
def test_get_with_parents_as_list(self):
    projects = self._create_projects_hierarchy()
    ref = projects[2]
    ref['parents_as_list'] = []
    for i in range(0, len(projects) - 1):
        ref['parents_as_list'].append(projects[i])
    self.stub_entity('GET', id=ref['id'], entity=ref)
    returned = self.manager.get(ref['id'], parents_as_list=True)
    self.assertQueryStringIs('parents_as_list')
    for i in range(0, len(projects) - 1):
        for attr in projects[i]:
            parent = getattr(returned, 'parents_as_list')[i]
            self.assertEqual(parent[attr], projects[i][attr], 'Expected different %s' % attr)