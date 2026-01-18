import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_delete_config_by_group(self):
    """Call ``DELETE /domains{domain_id}/config/{group}``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    self.delete('/domains/%(domain_id)s/config/ldap' % {'domain_id': self.domain['id']})
    res = PROVIDERS.domain_config_api.get_config(self.domain['id'])
    self.assertNotIn('ldap', res)