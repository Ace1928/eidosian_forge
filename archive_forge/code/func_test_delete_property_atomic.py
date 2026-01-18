from unittest import mock
from glance.domain import proxy
import glance.tests.utils as test_utils
def test_delete_property_atomic(self):
    image = mock.MagicMock()
    image.image_id = 'foo'
    self._test_method('delete_property_atomic', None, image, 'foo', 'bar')