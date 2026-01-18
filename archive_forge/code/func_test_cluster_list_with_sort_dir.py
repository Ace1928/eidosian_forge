import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_list_with_sort_dir(self):
    expect = [('GET', '/v1/clusters/?sort_dir=asc', {}, None)]
    self._test_cluster_list_with_filters(sort_dir='asc', expect=expect)