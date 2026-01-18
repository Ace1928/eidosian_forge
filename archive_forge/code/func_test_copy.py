import doctest
import unittest
from genshi.template.base import Template, Context
def test_copy(self):
    orig_ctxt = Context(a=5, b=6)
    orig_ctxt.push({'c': 7})
    orig_ctxt._match_templates.append(object())
    orig_ctxt._choice_stack.append(object())
    ctxt = orig_ctxt.copy()
    self.assertNotEqual(id(orig_ctxt), id(ctxt))
    self.assertEqual(repr(orig_ctxt), repr(ctxt))
    self.assertEqual(orig_ctxt._match_templates, ctxt._match_templates)
    self.assertEqual(orig_ctxt._choice_stack, ctxt._choice_stack)