import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_list_with_sort_key(self):
    expect = [('GET', '/v1/clusters/?sort_key=uuid', {}, None)]
    self._test_cluster_list_with_filters(sort_key='uuid', expect=expect)