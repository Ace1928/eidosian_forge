import os
from pathlib import Path
import unittest
from traits.api import File, HasTraits, TraitError
from traits.testing.optional_dependencies import requires_traitsui
def test_invalid_pathlike_file(self):
    example_model = ExampleModel(file_name=__file__)
    with self.assertRaises(TraitError):
        example_model.file_name = Path('not_valid_path!#!#!#')