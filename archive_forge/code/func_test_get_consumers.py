from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test_get_consumers(self):
    self.assertEqual(self.consumers, self.key.consumers)