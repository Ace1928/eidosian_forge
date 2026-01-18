from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_exception_not_found_with_image(self):
    msg = glance_store.NotFound(image='123')
    self.assertIn('Image 123 not found', encodeutils.exception_to_unicode(msg))