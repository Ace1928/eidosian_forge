import enum
import unittest
from traits.api import (
from traits.etsconfig.api import ETSConfig
from traits.testing.optional_dependencies import requires_traitsui
def test_invalid_enum(self):
    example_model = ExampleModel(root='model1')

    def assign_invalid():
        example_model.root = 'not_valid_model'
    self.assertRaises(TraitError, assign_invalid)