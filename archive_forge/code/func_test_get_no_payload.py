from unittest import mock
from openstack.key_manager.v1 import secret
from openstack.tests.unit import base
def test_get_no_payload(self):
    sot = secret.Secret(id='id')
    sess = mock.Mock()
    rv = mock.Mock()
    return_body = {'status': 'cool'}
    rv.json = mock.Mock(return_value=return_body)
    sess.get = mock.Mock(return_value=rv)
    sot.fetch(sess)
    sess.get.assert_called_once_with('secrets/id')