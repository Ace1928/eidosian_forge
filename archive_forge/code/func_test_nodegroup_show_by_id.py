import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_show_by_id(self):
    nodegroup = self.mgr.get(self.cluster_id, NODEGROUP1['id'])
    expect = [('GET', self.base_path + '%s' % NODEGROUP1['id'], {}, None)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NODEGROUP1['name'], nodegroup.name)