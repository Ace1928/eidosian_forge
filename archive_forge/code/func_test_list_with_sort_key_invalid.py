import testtools
from glanceclient.tests.unit.v2 import base
from glanceclient.tests import utils
from glanceclient.v2 import metadefs
def test_list_with_sort_key_invalid(self):
    self.assertRaises(ValueError, self.controller.list, sort_key='foo')