from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_import_error(self):
    self.assertRaises(ImportError, get_named_object, 'NO_SUCH_MODULE')