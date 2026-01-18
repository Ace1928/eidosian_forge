from castellan.key_manager import not_implemented_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_create_key(self):
    self.assertRaises(NotImplementedError, self.key_mgr.create_key, None)