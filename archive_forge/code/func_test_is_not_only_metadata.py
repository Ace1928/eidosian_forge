from castellan.common import objects
from castellan.common.objects import private_key
from castellan.tests import base
from castellan.tests import utils
def test_is_not_only_metadata(self):
    self.assertFalse(self.key.is_metadata_only())