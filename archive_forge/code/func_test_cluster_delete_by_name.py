import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_delete_by_name(self):
    cluster = self.mgr.delete(CLUSTER1['name'])
    expect = [('DELETE', '/v1/clusters/%s' % CLUSTER1['name'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertIsNone(cluster)