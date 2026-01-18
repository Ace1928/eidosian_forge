from unittest import mock
from openstack.compute.v2 import service
from openstack import exceptions
from openstack.tests.unit import base
def test_find_multiple_match(self):
    data = [service.Service(name='bin1', host='host', id=1), service.Service(name='bin1', host='host', id=2)]
    with mock.patch.object(service.Service, 'list') as list_mock:
        list_mock.return_value = data
        self.assertRaises(exceptions.DuplicateResource, service.Service.find, self.sess, 'bin1', ignore_missing=False, host='host')