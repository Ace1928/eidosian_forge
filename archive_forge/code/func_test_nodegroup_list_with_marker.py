import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list_with_marker(self):
    filter_ = '?marker=%s' % NODEGROUP2['uuid']
    expect = [('GET', self.base_path + filter_, {}, None)]
    self._test_nodegroup_list_with_filters(self.cluster_id, marker=NODEGROUP2['uuid'], expect=expect)