from cryptography import exceptions as crypto_exception
from cursive import exception as cursive_exception
from cursive import signature_utils
import glance_store
from unittest import mock
from glance.common import exception
import glance.location
from glance.tests.unit import base as unit_test_base
from glance.tests.unit import utils as unit_test_utils
from glance.tests import utils
def test_new_image_with_location(self):
    locations = [{'url': '%s/%s' % (BASE_URI, UUID1), 'metadata': {}}]
    image = self.image_factory.new_image(locations=locations)
    self.assertEqual(locations, image.locations)
    location_bad = {'url': 'unknown://location', 'metadata': {}}
    self.assertRaises(exception.BadStoreUri, self.image_factory.new_image, locations=[location_bad])