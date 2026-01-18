import uuid
import http.client
from testtools import matchers
from keystone.common import provider_api
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import test_v3
from keystone.tests.unit import utils as test_utils
def test_create_project_with_parent_id_none_and_domain_id_none(self):
    """Call ``POST /projects``."""
    collection_url = '/domains/%(domain_id)s/users/%(user_id)s/roles' % {'domain_id': self.domain_id, 'user_id': self.user['id']}
    member_url = '%(collection_url)s/%(role_id)s' % {'collection_url': collection_url, 'role_id': self.role_id}
    self.put(member_url)
    auth = self.build_authentication_request(user_id=self.user['id'], password=self.user['password'], domain_id=self.domain_id)
    ref = unit.new_project_ref()
    r = self.post('/projects', auth=auth, body={'project': ref})
    ref['domain_id'] = self.domain['id']
    self.assertValidProjectResponse(r, ref)