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
def test_create_second_domain_valid_explicit_id_fails(self):
    """Call ``POST /domains`` with a valid `explicit_domain_id` set."""
    ref = unit.new_domain_ref()
    explicit_domain_id = '9aea63518f0040c6b4518d8d2242911c'
    ref['explicit_domain_id'] = explicit_domain_id
    r = self.post('/domains', body={'domain': ref})
    self.assertValidDomainResponse(r, ref)
    r = self.post('/domains', body={'domain': ref}, expected_status=http.client.CONFLICT)