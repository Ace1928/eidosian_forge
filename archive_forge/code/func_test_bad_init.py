import unittest
import uuid
from traits.api import HasTraits, TraitError, UUID
def test_bad_init(self):
    with self.assertRaises(TraitError):
        A(id=uuid.uuid4())