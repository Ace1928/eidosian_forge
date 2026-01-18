import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_non_existant_config_group(self):
    """Call ``GET /domains/{domain_id}/config/{group_not_exist}``."""
    config = {'ldap': {'url': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    url = '/domains/%(domain_id)s/config/identity' % {'domain_id': self.domain['id']}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)