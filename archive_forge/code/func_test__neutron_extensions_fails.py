import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test__neutron_extensions_fails(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), status_code=404)])
    with testtools.ExpectedException(exceptions.ResourceNotFound):
        self.cloud._neutron_extensions()
    self.assert_calls()