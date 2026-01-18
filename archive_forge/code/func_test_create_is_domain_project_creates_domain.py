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
def test_create_is_domain_project_creates_domain(self):
    """Call ``POST /projects`` is_domain and check a domain is created."""
    project_ref = unit.new_project_ref(domain_id=None, is_domain=True)
    r = self.post('/projects', body={'project': project_ref})
    self.assertValidProjectResponse(r)
    r = self.get('/domains/%(domain_id)s' % {'domain_id': r.result['project']['id']})
    self.assertValidDomainResponse(r)
    self.assertIsNotNone(r.result['domain'])