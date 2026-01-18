from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_locateNodes(self):
    doc1 = self.dom.parseString('<a><b foo="olive"><c foo="olive"/></b><d foo="poopy"/></a>')
    doc = self.dom.Document()
    node_list = domhelpers.locateNodes(doc1.childNodes, 'foo', 'olive', noNesting=1)
    actual = ''.join([node.toxml() for node in node_list])
    expected = doc.createElement('b')
    expected.setAttribute('foo', 'olive')
    c = doc.createElement('c')
    c.setAttribute('foo', 'olive')
    expected.appendChild(c)
    self.assertEqual(actual, expected.toxml())
    node_list = domhelpers.locateNodes(doc1.childNodes, 'foo', 'olive', noNesting=0)
    actual = ''.join([node.toxml() for node in node_list])
    self.assertEqual(actual, expected.toxml() + c.toxml())