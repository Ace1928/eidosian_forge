from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test_is_only_metadata(self):
    k = private_key.PrivateKey(self.algorithm, self.bit_length, None, self.name, self.created)
    self.assertTrue(k.is_metadata_only())