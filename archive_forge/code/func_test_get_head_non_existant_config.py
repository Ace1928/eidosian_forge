import copy
import uuid
import http.client
from keystone.common import provider_api
import keystone.conf
from keystone import exception
from keystone.tests import unit
from keystone.tests.unit import test_v3
def test_get_head_non_existant_config(self):
    """Call ``GET /domains{domain_id}/config when no config defined``."""
    url = '/domains/%(domain_id)s/config' % {'domain_id': self.domain['id']}
    self.get(url, expected_status=http.client.NOT_FOUND)
    self.head(url, expected_status=http.client.NOT_FOUND)