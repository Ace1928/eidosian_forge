import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create_with_keypair(self):
    cluster_with_keypair = dict()
    cluster_with_keypair.update(CREATE_CLUSTER)
    cluster_with_keypair['keypair'] = 'test_key'
    cluster = self.mgr.create(**cluster_with_keypair)
    expect = [('POST', '/v1/clusters', {}, cluster_with_keypair)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster)