from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_gatherTextNodes(self):
    doc1 = self.dom.parseString('<a>foo</a>')
    actual = domhelpers.gatherTextNodes(doc1)
    expected = 'foo'
    self.assertEqual(actual, expected)
    actual = domhelpers.gatherTextNodes(doc1.documentElement)
    self.assertEqual(actual, expected)
    doc2_xml = '<a>a<b>b</b><c>c</c>def<g>g<h>h</h></g></a>'
    doc2 = self.dom.parseString(doc2_xml)
    actual = domhelpers.gatherTextNodes(doc2)
    expected = 'abcdefgh'
    self.assertEqual(actual, expected)
    actual = domhelpers.gatherTextNodes(doc2.documentElement)
    self.assertEqual(actual, expected)
    doc3_xml = '<a>a<b>b<d>d<g>g</g><h>h</h></d><e>e<i>i</i></e></b>' + '<c>c<f>f<j>j</j></f></c></a>'
    doc3 = self.dom.parseString(doc3_xml)
    actual = domhelpers.gatherTextNodes(doc3)
    expected = 'abdgheicfj'
    self.assertEqual(actual, expected)
    actual = domhelpers.gatherTextNodes(doc3.documentElement)
    self.assertEqual(actual, expected)