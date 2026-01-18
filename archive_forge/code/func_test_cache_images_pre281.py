from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import aggregates as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import aggregates
from novaclient.v2 import images
def test_cache_images_pre281(self):
    self.cs.api_version = api_versions.APIVersion('2.80')
    aggregate = self.cs.aggregates.list()[0]
    _images = [images.Image(self.cs.aggregates, {'id': '1'})]
    self.assertRaises(exceptions.VersionNotFoundForAPIMethod, aggregate.cache_images, _images)