import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_list(self):
    clusters = self.mgr.list()
    expect = [('GET', '/v1/clusters', {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertThat(clusters, matchers.HasLength(2))