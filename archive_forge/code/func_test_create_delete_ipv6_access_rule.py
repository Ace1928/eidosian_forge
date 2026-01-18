import ast
import ddt
from tempest.lib import exceptions as tempest_lib_exc
from manilaclient import api_versions
from manilaclient import config
from manilaclient.tests.functional import base
@ddt.data('2.38', api_versions.MAX_VERSION)
def test_create_delete_ipv6_access_rule(self, microversion):
    self._create_delete_access_rule(self.share_id, 'ip', self.access_to['ipv6'].pop(), microversion)