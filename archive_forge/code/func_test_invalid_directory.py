import pathlib
from tempfile import gettempdir
import unittest
from traits.api import BaseDirectory, Directory, HasTraits, TraitError
def test_invalid_directory(self):
    example_model = ExampleModel(path=gettempdir())

    def assign_invalid():
        example_model.path = 'not_valid_path!#!#!#'
    self.assertRaises(TraitError, assign_invalid)