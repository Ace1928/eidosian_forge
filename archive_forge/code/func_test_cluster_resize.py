import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_resize(self):
    body = {'node_count': RESIZED_NODE_COUNT}
    cluster = self.mgr.resize(CLUSTER1['uuid'], **body)
    expect = [('POST', '/v1/clusters/%s/actions/resize' % CLUSTER1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(RESIZED_NODE_COUNT, cluster.node_count)