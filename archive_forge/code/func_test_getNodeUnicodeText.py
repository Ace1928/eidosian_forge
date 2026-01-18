from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_getNodeUnicodeText(self):
    """
        L{domhelpers.getNodeText} returns a C{unicode} string when text
        nodes are represented in the DOM with unicode, whether or not there
        are non-ASCII characters present.
        """
    node = self.dom.parseString('<foo>bar</foo>')
    text = domhelpers.getNodeText(node)
    self.assertEqual(text, 'bar')
    self.assertIsInstance(text, str)
    node = self.dom.parseString('<foo>☃</foo>'.encode())
    text = domhelpers.getNodeText(node)
    self.assertEqual(text, '☃')
    self.assertIsInstance(text, str)