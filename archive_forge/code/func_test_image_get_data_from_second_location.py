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
def test_image_get_data_from_second_location(self):

    def fake_get_from_backend(self, location, offset=0, chunk_size=None, context=None):
        if UUID1 in location:
            raise Exception('not allow download from %s' % location)
        else:
            return self.data[location]
    image1 = glance.location.ImageProxy(self.image_stub, {}, self.store_api, self.store_utils)
    self.assertEqual('XXX', image1.get_data())
    context = glance.context.RequestContext(user=USER1)
    image2, image_stub2 = self._add_image(context, UUID2, 'ZZZ', 3)
    location_data = image2.locations[0]
    with mock.patch('glance.location.store') as mock_store:
        mock_store.get_size_from_uri_and_backend.return_value = 3
        image1.locations.append(location_data)
    self.assertEqual(2, len(image1.locations))
    self.assertEqual(UUID2, location_data['url'])
    self.mock_object(unit_test_utils.FakeStoreAPI, 'get_from_backend', fake_get_from_backend)
    self.assertEqual('ZZZ', image1.get_data().data.fd._source)
    image1.locations.pop(0)
    self.assertEqual(1, len(image1.locations))
    image2.delete()