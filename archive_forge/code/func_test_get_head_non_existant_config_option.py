import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_non_existant_config_option(self):
    """Test that Not Found is returned when option doesn't exist.

        Call ``GET & HEAD /domains/{domain_id}/config/{group}/{opt_not_exist}``
        and ensure a Not Found is returned because the option isn't defined
        within the group.
        """
    config = {'ldap': {'url': uuid.uuid4().hex}}
    PROVIDERS.domain_config_api.create_config(self.domain['id'], config)
    url = '/domains/%(domain_id)s/config/ldap/user_tree_dn' % {'domain_id': self.domain['id']}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)