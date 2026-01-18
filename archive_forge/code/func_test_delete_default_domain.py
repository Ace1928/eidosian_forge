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
def test_delete_default_domain(self):
    self.patch('/domains/%(domain_id)s' % {'domain_id': CONF.identity.default_domain_id}, body={'domain': {'enabled': False}})
    self.delete('/domains/%(domain_id)s' % {'domain_id': CONF.identity.default_domain_id})