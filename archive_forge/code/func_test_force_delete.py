from unittest import mock
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def test_force_delete(self):
    sot = cluster.Cluster(**FAKE)
    resp = mock.Mock()
    fake_action_id = 'f1de9847-2382-4272-8e73-cab0bc194663'
    resp.headers = {'Location': fake_action_id}
    resp.json = mock.Mock(return_value={'foo': 'bar'})
    resp.status_code = 200
    sess = mock.Mock()
    sess.delete = mock.Mock(return_value=resp)
    res = sot.force_delete(sess)
    self.assertEqual(fake_action_id, res.id)
    url = 'clusters/%s' % sot.id
    body = {'force': True}
    sess.delete.assert_called_once_with(url, json=body)