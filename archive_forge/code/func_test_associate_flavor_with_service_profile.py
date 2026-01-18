from unittest import mock
from openstack.network.v2 import flavor
from openstack.tests.unit import base
def test_associate_flavor_with_service_profile(self):
    flav = flavor.Flavor(EXAMPLE)
    response = mock.Mock()
    response.body = {'service_profile': {'id': '1'}}
    response.json = mock.Mock(return_value=response.body)
    sess = mock.Mock()
    sess.post = mock.Mock(return_value=response)
    flav.id = 'IDENTIFIER'
    self.assertEqual(response.body, flav.associate_flavor_with_service_profile(sess, '1'))
    url = 'flavors/IDENTIFIER/service_profiles'
    sess.post.assert_called_with(url, json=response.body)