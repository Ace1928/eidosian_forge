import unittest
from traits.adaptation.api import AdaptationManager
import traits.adaptation.tests.abc_examples
import traits.adaptation.tests.interface_examples
def test_adaptation_prefers_subclasses_other_registration_order(self):
    ex = self.examples
    self.adaptation_manager.register_factory(factory=ex.EditorToIPrintable, from_protocol=ex.Editor, to_protocol=ex.IPrintable)
    self.adaptation_manager.register_factory(factory=ex.TextEditorToIPrintable, from_protocol=ex.TextEditor, to_protocol=ex.IPrintable)
    text_editor = ex.TextEditor()
    printable = self.adaptation_manager.adapt(text_editor, ex.IPrintable)
    self.assertIsNotNone(printable)
    self.assertIs(type(printable), ex.TextEditorToIPrintable)