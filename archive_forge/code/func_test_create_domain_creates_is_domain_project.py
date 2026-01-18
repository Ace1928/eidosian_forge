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
def test_create_domain_creates_is_domain_project(self):
    """Check a project that acts as a domain is created.

        Call ``POST /domains``.
        """
    domain_ref = unit.new_domain_ref()
    r = self.post('/domains', body={'domain': domain_ref})
    self.assertValidDomainResponse(r, domain_ref)
    r = self.get('/projects/%(project_id)s' % {'project_id': r.result['domain']['id']})
    self.assertValidProjectResponse(r)
    self.assertTrue(r.result['project']['is_domain'])
    self.assertIsNone(r.result['project']['parent_id'])
    self.assertIsNone(r.result['project']['domain_id'])