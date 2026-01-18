import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create(self):
    cluster = self.mgr.create(**CREATE_CLUSTER)
    expect = [('POST', '/v1/clusters', {}, CREATE_CLUSTER)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster)