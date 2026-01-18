import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_create_with_docker_volume_size(self):
    ng_with_volume_size = dict()
    ng_with_volume_size.update(CREATE_NODEGROUP)
    ng_with_volume_size['docker_volume_size'] = 20
    nodegroup = self.mgr.create(self.cluster_id, **ng_with_volume_size)
    expect = [('POST', self.base_path, {}, ng_with_volume_size)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(nodegroup)