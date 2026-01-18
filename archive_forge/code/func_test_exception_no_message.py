from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_exception_no_message(self):
    msg = glance_store.NotFound()
    self.assertIn('Image %(image)s not found', encodeutils.exception_to_unicode(msg))