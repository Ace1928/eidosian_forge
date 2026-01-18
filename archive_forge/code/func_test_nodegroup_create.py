import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_create(self):
    nodegroup = self.mgr.create(self.cluster_id, **CREATE_NODEGROUP)
    expect = [('POST', self.base_path, {}, CREATE_NODEGROUP)]
    self.assertEqual(expect, self.api.calls)
    self.assertTrue(nodegroup)