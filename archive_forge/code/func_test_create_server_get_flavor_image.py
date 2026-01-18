import base64
from unittest import mock
import uuid
from openstack.cloud import meta
from openstack.compute.v2 import server
from openstack import connection
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_server_get_flavor_image(self):
    self.use_glance()
    image_id = str(uuid.uuid4())
    fake_image_dict = fakes.make_fake_image(image_id=image_id)
    fake_image_search_return = {'images': [fake_image_dict]}
    build_server = fakes.make_fake_server('1234', '', 'BUILD')
    active_server = fakes.make_fake_server('1234', '', 'BUILD')
    self.register_uris([dict(method='GET', uri='https://image.example.com/v2/images', json=fake_image_search_return), self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['flavors', 'vanilla'], qs_elements=[]), json=fakes.FAKE_FLAVOR), dict(method='POST', uri=self.get_mock_url('compute', 'public', append=['servers']), json={'server': build_server}, validate=dict(json={'server': {'flavorRef': fakes.FLAVOR_ID, 'imageRef': image_id, 'max_count': 1, 'min_count': 1, 'networks': [{'uuid': 'some-network'}], 'name': 'server-name'}})), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), json={'server': active_server}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': []})])
    self.cloud.create_server('server-name', image_id, 'vanilla', nics=[{'net-id': 'some-network'}], wait=False)
    self.assert_calls()