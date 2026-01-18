import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_config_by_option(self):
    """Call ``GET & HEAD /domains{domain_id}/config/{group}/{option}``."""
    PROVIDERS.domain_config_api.create_config(self.domain['id'], self.config)
    url = '/domains/%(domain_id)s/config/ldap/url' % {'domain_id': self.domain['id']}
    r = self.get(url)
    self.assertEqual({'url': self.config['ldap']['url']}, r.result['config'])
    self.head(url, expected_status=http.client.OK)