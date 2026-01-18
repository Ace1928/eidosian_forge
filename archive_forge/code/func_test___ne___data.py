from castellan.common import objects
from castellan.common.objects import passphrase
from castellan.tests import base
def test___ne___data(self):
    other_phrase = passphrase.Passphrase(b'other passphrase', self.name)
    self.assertTrue(self.passphrase != other_phrase)