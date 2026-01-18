import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_list_with_marker(self):
    expect = [('GET', '/v1/clusters/?marker=%s' % CLUSTER2['uuid'], {}, None)]
    self._test_cluster_list_with_filters(marker=CLUSTER2['uuid'], expect=expect)