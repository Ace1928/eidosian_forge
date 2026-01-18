import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create_with_cluster_create_timeout(self):
    cluster_with_timeout = dict()
    cluster_with_timeout.update(CREATE_CLUSTER)
    cluster_with_timeout['create_timeout'] = '15'
    cluster = self.mgr.create(**cluster_with_timeout)
    expect = [('POST', '/v1/clusters', {}, cluster_with_timeout)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster)