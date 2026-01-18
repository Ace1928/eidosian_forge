import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test__has_nova_extension(self):
    body = [{'updated': '2014-12-03T00:00:00Z', 'name': 'Multinic', 'links': [], 'namespace': 'http://openstack.org/compute/ext/fake_xml', 'alias': 'NMN', 'description': 'Multiple network support.'}, {'updated': '2014-12-03T00:00:00Z', 'name': 'DiskConfig', 'links': [], 'namespace': 'http://openstack.org/compute/ext/fake_xml', 'alias': 'OS-DCF', 'description': 'Disk Management Extension.'}]
    self.register_uris([dict(method='GET', uri='{endpoint}/extensions'.format(endpoint=fakes.COMPUTE_ENDPOINT), json=dict(extensions=body))])
    self.assertTrue(self.cloud._has_nova_extension('NMN'))
    self.assert_calls()