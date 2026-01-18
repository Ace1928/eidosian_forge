from unittest import mock
import uuid
from openstack.load_balancer.v2 import load_balancer
from openstack.tests.unit import base
def test_delete_cascade(self):
    sess = mock.Mock()
    resp = mock.Mock()
    sess.delete.return_value = resp
    sot = load_balancer.LoadBalancer(**EXAMPLE)
    sot.cascade = True
    sot._translate_response = mock.Mock()
    sot.delete(sess)
    url = 'lbaas/loadbalancers/%(lb)s' % {'lb': EXAMPLE['id']}
    params = {'cascade': True}
    sess.delete.assert_called_with(url, params=params)
    sot._translate_response.assert_called_once_with(resp, error_message=None, has_body=False)