import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
def test_cluster_update_with_rollback(self):
    patch = {'op': 'replace', 'value': NEW_NAME, 'path': '/name'}
    cluster = self.mgr.update(id=CLUSTER1['id'], patch=patch, rollback=True)
    expect = [('PATCH', '/v1/clusters/%s/?rollback=True' % CLUSTER1['id'], {}, patch)]
    self.assertEqual(expect, self.api.calls)
    self.assertEqual(NEW_NAME, cluster.name)