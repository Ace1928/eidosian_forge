import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import mservices
def test_coe_service_list(self):
    mservices = self.mgr.list()
    expect = [('GET', '/v1/mservices', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(mservices, matchers.HasLength(2))