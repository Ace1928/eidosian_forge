from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_clearNode(self):
    doc1 = self.dom.parseString('<a><b><c><d/></c></b></a>')
    a_node = doc1.documentElement
    domhelpers.clearNode(a_node)
    self.assertEqual(a_node.toxml(), self.dom.Element('a').toxml())
    doc2 = self.dom.parseString('<a><b><c><d/></c></b></a>')
    b_node = doc2.documentElement.childNodes[0]
    domhelpers.clearNode(b_node)
    actual = doc2.documentElement.toxml()
    expected = self.dom.Element('a')
    expected.appendChild(self.dom.Element('b'))
    self.assertEqual(actual, expected.toxml())