from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_getElementsByTagName(self):
    doc1 = self.dom.parseString('<foo/>')
    actual = domhelpers.getElementsByTagName(doc1, 'foo')[0].nodeName
    expected = 'foo'
    self.assertEqual(actual, expected)
    el1 = doc1.documentElement
    actual = domhelpers.getElementsByTagName(el1, 'foo')[0].nodeName
    self.assertEqual(actual, expected)
    doc2_xml = '<a><foo in="a"/><b><foo in="b"/></b><c><foo in="c"/></c><foo in="d"/><foo in="ef"/><g><foo in="g"/><h><foo in="h"/></h></g></a>'
    doc2 = self.dom.parseString(doc2_xml)
    tag_list = domhelpers.getElementsByTagName(doc2, 'foo')
    actual = ''.join([node.getAttribute('in') for node in tag_list])
    expected = 'abcdefgh'
    self.assertEqual(actual, expected)
    el2 = doc2.documentElement
    tag_list = domhelpers.getElementsByTagName(el2, 'foo')
    actual = ''.join([node.getAttribute('in') for node in tag_list])
    self.assertEqual(actual, expected)
    doc3_xml = '\n<a><foo in="a"/>\n    <b><foo in="b"/>\n        <d><foo in="d"/>\n            <g><foo in="g"/></g>\n            <h><foo in="h"/></h>\n        </d>\n        <e><foo in="e"/>\n            <i><foo in="i"/></i>\n        </e>\n    </b>\n    <c><foo in="c"/>\n        <f><foo in="f"/>\n            <j><foo in="j"/></j>\n        </f>\n    </c>\n</a>'
    doc3 = self.dom.parseString(doc3_xml)
    tag_list = domhelpers.getElementsByTagName(doc3, 'foo')
    actual = ''.join([node.getAttribute('in') for node in tag_list])
    expected = 'abdgheicfj'
    self.assertEqual(actual, expected)
    el3 = doc3.documentElement
    tag_list = domhelpers.getElementsByTagName(el3, 'foo')
    actual = ''.join([node.getAttribute('in') for node in tag_list])
    self.assertEqual(actual, expected)
    doc4_xml = '<foo><bar></bar><baz><foo/></baz></foo>'
    doc4 = self.dom.parseString(doc4_xml)
    actual = domhelpers.getElementsByTagName(doc4, 'foo')
    root = doc4.documentElement
    expected = [root, root.childNodes[-1].childNodes[0]]
    self.assertEqual(actual, expected)
    actual = domhelpers.getElementsByTagName(root, 'foo')
    self.assertEqual(actual, expected)