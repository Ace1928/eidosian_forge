import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def test_coe_service_list_with_sort_dir(self):
    expect = [('GET', '/v1/mservices/?sort_dir=asc', {}, None)]
    self._test_coe_service_list_with_filters(sort_dir='asc', expect=expect)