import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def test_coe_service_list_with_marker(self):
    expect = [('GET', '/v1/mservices/?marker=%s' % SERVICE2['id'], {}, None)]
    self._test_coe_service_list_with_filters(marker=SERVICE2['id'], expect=expect)