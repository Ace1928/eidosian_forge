import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def test_coe_service_list_with_limit(self):
    expect = [('GET', '/v1/mservices/?limit=2', {}, None)]
    self._test_coe_service_list_with_filters(limit=2, expect=expect)