import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
@requires_traitsui
def test_get_editor(self):
    obj = HasTimeTraits()
    trait = obj.base_trait('epoch')
    editor_factory = trait.get_editor()
    self.assertIsInstance(editor_factory, traitsui.api.TimeEditor)