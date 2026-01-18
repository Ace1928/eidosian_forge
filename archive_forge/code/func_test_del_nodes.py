from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_del_nodes(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    resp.json = mock.Mock(return_value='')
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=resp)
    self.assertEqual('', sot.del_nodes(sess, ['node-11']))
    url = 'clusters/%s/actions' % sot.id
    body = {'del_nodes': {'nodes': ['node-11']}}
    sess.post.assert_called_once_with(url, json=body)