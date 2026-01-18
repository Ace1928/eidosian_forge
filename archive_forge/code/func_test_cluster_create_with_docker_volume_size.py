import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_create_with_docker_volume_size(self):
    cluster_with_volume_size = dict()
    cluster_with_volume_size.update(CREATE_CLUSTER)
    cluster_with_volume_size['docker_volume_size'] = 20
    cluster = self.mgr.create(**cluster_with_volume_size)
    expect = [('POST', '/v1/clusters', {}, cluster_with_volume_size)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(cluster)