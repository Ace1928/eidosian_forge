from breezy import branch, tests
from breezy.pyutils import calc_parent_name, get_named_object
def test_attribute_error(self):
    self.assertRaises(AttributeError, get_named_object, 'sys', 'NO_SUCH_ATTR')