from oslo_utils import encodeutils
from oslotest import base
import glance_store
def test_exception_with_message(self):
    msg = glance_store.NotFound('Some message')
    self.assertIn('Some message', encodeutils.exception_to_unicode(msg))