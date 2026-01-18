from unittest import mock
from keystoneauth1 import adapter
from openstack.shared_file_system.v2 import share
from openstack.tests.unit import base
def test_manage_share(self):
    sot = share.Share()
    self.resp.headers = {}
    self.resp.json = mock.Mock(return_value={'share': {'name': 'test_share', 'size': 1}})
    export_path = '10.254.0 .5:/shares/share-42033c24-0261-424f-abda-4fef2f6dbfd5.'
    params = {'name': 'test_share'}
    res = sot.manage(self.sess, sot['share_protocol'], export_path, sot['host'], **params)
    self.assertEqual(res.name, 'test_share')
    self.assertEqual(res.size, 1)
    jsonDict = {'share': {'protocol': sot['share_protocol'], 'export_path': export_path, 'service_host': sot['host'], 'name': 'test_share'}}
    self.sess.post.assert_called_once_with('shares/manage', json=jsonDict)