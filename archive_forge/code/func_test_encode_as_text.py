from oslo_serialization import base64
from oslotest import base as test_base
def test_encode_as_text(self):
    self.assertEqual('dGV4dA==', base64.encode_as_text(b'text'))
    self.assertEqual('dGV4dA==', base64.encode_as_text('text'))
    self.assertEqual('ZTrDqQ==', base64.encode_as_text('e:é'))
    self.assertEqual('ZTrp', base64.encode_as_text('e:é', encoding='latin1'))