from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import aggregates as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import aggregates
from novaclient.v2 import images
def test_cache_images(self):
    aggregate = self.cs.aggregates.list()[0]
    _images = [images.Image(self.cs.aggregates, {'id': '1'}), images.Image(self.cs.aggregates, {'id': '2'})]
    aggregate.cache_images(_images)
    expected_body = {'cache': [{'id': image.id} for image in _images]}
    self.assert_called('POST', '/os-aggregates/1/images', expected_body)