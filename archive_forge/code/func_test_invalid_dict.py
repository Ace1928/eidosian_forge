from castellan.common import exception
from castellan.common import objects
from castellan.tests import base
def test_invalid_dict(self):
    self.assertRaises(exception.InvalidManagedObjectDictError, objects.from_dict, {})