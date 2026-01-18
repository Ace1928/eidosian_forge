import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_upgrade(self):
    body = {'cluster_template': UPGRADED_TO_TEMPLATE, 'max_batch_size': 1}
    cluster = self.mgr.upgrade(CLUSTER1['uuid'], **body)
    expect = [('POST', '/v1/clusters/%s/actions/upgrade' % CLUSTER1['uuid'], {}, body)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(UPGRADED_TO_TEMPLATE, cluster.cluster_template_id)