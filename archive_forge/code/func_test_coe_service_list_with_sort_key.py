import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def test_coe_service_list_with_sort_key(self):
    expect = [('GET', '/v1/mservices/?sort_key=id', {}, None)]
    self._test_coe_service_list_with_filters(sort_key='id', expect=expect)