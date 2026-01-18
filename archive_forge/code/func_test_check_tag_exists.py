from unittest import mock
from keystoneauth1 import adapter
from openstack.common import tag
from openstack import exceptions
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit.test_resource import FakeResponse
def test_check_tag_exists(self):
    res = self.sot
    sess = self.session
    sess.get.side_effect = [FakeResponse(None, 202)]
    result = res.check_tag(sess, 'blue')
    self.assertEqual([], res.tags)
    self.assertEqual(res, result)
    url = self.base_path + '/' + res.id + '/tags/blue'
    sess.get.assert_called_once_with(url)