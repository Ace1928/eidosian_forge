from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_children(self) -> None:
    s = '<foo><bar /><baz /><bax>foo</bax></foo>'
    d = microdom.parseString(s).documentElement
    self.assertEqual([n.nodeName for n in d.childNodes], ['bar', 'baz', 'bax'])
    self.assertEqual(d.lastChild().nodeName, 'bax')
    self.assertEqual(d.firstChild().nodeName, 'bar')
    self.assertTrue(d.hasChildNodes())
    self.assertTrue(not d.firstChild().hasChildNodes())