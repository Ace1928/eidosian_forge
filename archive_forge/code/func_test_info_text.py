import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_info_text(self):
    example_model = ExampleModel()
    with self.assertRaises(TraitError) as exc_cm:
        example_model.path = 47
    self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
    self.assertIn('referring to an existing directory', str(exc_cm.exception))
    with self.assertRaises(TraitError) as exc_cm:
        example_model.new_path = 47
    self.assertIn('a string or os.PathLike object', str(exc_cm.exception))
    self.assertNotIn('exist', str(exc_cm.exception))