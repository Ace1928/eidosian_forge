import copy
import testtools
from testtools import matchers
from magnumclient.tests import utils
from magnumclient.v1 import quotas
def test_list_quotas_all(self):
    quotas = self.mgr.list(all_tenants=True)
    expect = [('GET', '/v1/quotas?all_tenants=True', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(quotas, matchers.HasLength(2))