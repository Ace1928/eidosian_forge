from __future__ import annotations
from importlib import reload
from io import BytesIO
from typing_extensions import Literal
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom, sux
def test_entities(self) -> None:
    nodes = microdom.parseString('<b>&amp;&#12AB;</b>').documentElement.childNodes
    self.assertEqual(len(nodes), 2)
    self.assertEqual(nodes[0].data, '&amp;')
    self.assertEqual(nodes[1].data, '&#12AB;')
    self.assertEqual(nodes[0].cloneNode().toxml(), '&amp;')
    for n in nodes:
        self.assertTrue(isinstance(n, microdom.EntityReference))