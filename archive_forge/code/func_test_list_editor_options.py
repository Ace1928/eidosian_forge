import datetime
import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Instance, List, Str
from traits.editor_factories import (
from traits.testing.optional_dependencies import requires_traitsui, traitsui
def test_list_editor_options(self):
    trait = List(Str, rows=10, use_notebook=True, page_name='page')
    editor = list_editor(trait, trait)
    self.assertIsInstance(editor, traitsui.api.ListEditor)
    self.assertEqual(editor.trait_handler, trait)
    self.assertEqual(editor.rows, 10)
    self.assertTrue(editor.use_notebook)
    self.assertEqual(editor.page_name, 'page')