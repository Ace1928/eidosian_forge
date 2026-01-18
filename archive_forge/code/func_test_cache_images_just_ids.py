from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import aggregates as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import aggregates
from novaclient.v2 import images
def test_cache_images_just_ids(self):
    aggregate = self.cs.aggregates.list()[0]
    _images = ['1']
    aggregate.cache_images(_images)
    expected_body = {'cache': [{'id': '1'}]}
    self.assert_called('POST', '/os-aggregates/1/images', expected_body)