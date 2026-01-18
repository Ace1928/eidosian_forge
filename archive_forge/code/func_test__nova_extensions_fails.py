import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test__nova_extensions_fails(self):
    self.register_uris([dict(method='GET', uri='{endpoint}/extensions'.format(endpoint=fakes.COMPUTE_ENDPOINT), status_code=404)])
    self.assertRaises(exceptions.ResourceNotFound, self.cloud._nova_extensions)
    self.assert_calls()