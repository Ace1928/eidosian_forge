import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
def test_nodegroup_list_with_sort_key(self):
    expect = [('GET', '/v1/clusters/test/nodegroups/?sort_key=uuid', {}, None)]
    self._test_nodegroup_list_with_filters(self.cluster_id, sort_key='uuid', expect=expect)