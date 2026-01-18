import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_list_editor_list_instance(self):
    trait = List(Instance(HasTraits))
    editor = list_editor(trait, trait)
    self.assertIsInstance(editor, traitsui.api.TableEditor)