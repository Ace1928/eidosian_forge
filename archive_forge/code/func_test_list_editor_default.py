import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_list_editor_default(self):
    trait = List(Str)
    editor = list_editor(trait, trait)
    self.assertIsInstance(editor, traitsui.api.ListEditor)
    self.assertEqual(editor.trait_handler, trait)
    self.assertEqual(editor.rows, 5)
    self.assertFalse(editor.use_notebook)
    self.assertEqual(editor.page_name, '')