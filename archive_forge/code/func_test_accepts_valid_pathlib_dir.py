import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_accepts_valid_pathlib_dir(self):
    foo = ExistsBaseDirectory()
    foo.path = pathlib.Path(gettempdir())
    self.assertIsInstance(foo.path, str)