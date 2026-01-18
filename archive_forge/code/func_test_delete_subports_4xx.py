from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import trunk
from openstack.tests.unit import base
def test_delete_subports_4xx(self):
    sot = trunk.Trunk(**EXAMPLE)
    response = mock.Mock()
    msg = '.*borked'
    response.body = {'NeutronError': {'message': msg}}
    response.json = mock.Mock(return_value=response.body)
    response.ok = False
    response.status_code = 404
    response.headers = {'content-type': 'application/json'}
    sess = mock.Mock()
    sess.put = mock.Mock(return_value=response)
    subports = [{'port_id': 'abc', 'segmentation_id': '123', 'segmentation_type': 'vlan'}]
    with testtools.ExpectedException(exceptions.ResourceNotFound, msg):
        sot.delete_subports(sess, subports)