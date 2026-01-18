from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_getIfExists(self):
    doc1 = self.dom.parseString('<a><b id="bar"/><c class="foo"/></a>')
    doc = self.dom.Document()
    node = domhelpers.getIfExists(doc1, 'foo')
    actual = node.toxml()
    expected = doc.createElement('c')
    expected.setAttribute('class', 'foo')
    self.assertEqual(actual, expected.toxml())
    node = domhelpers.getIfExists(doc1, 'pzork')
    self.assertIdentical(node, None)