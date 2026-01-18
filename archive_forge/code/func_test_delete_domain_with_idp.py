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
def test_delete_domain_with_idp(self):
    domain_ref = unit.new_domain_ref()
    r = self.post('/domains', body={'domain': domain_ref})
    self.assertValidDomainResponse(r, domain_ref)
    domain_id = r.result['domain']['id']
    self.put('/OS-FEDERATION/identity_providers/test_idp', body={'identity_provider': {'domain_id': domain_id}}, expected_status=http.client.CREATED)
    self.patch('/domains/%(domain_id)s' % {'domain_id': domain_id}, body={'domain': {'enabled': False}})
    self.delete('/domains/%s' % domain_id)
    self.get('/OS-FEDERATION/identity_providers/test_idp', expected_status=http.client.NOT_FOUND)