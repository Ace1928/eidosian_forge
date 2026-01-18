from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test___ne___algorithm(self):
    other_key = private_key.PrivateKey('DSA', self.bit_length, self.encoded, self.name, consumers=self.consumers)
    self.assertTrue(self.key != other_key)