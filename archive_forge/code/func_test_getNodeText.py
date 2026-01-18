from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_getNodeText(self):
    """
        L{getNodeText} returns the concatenation of all the text data at or
        beneath the node passed to it.
        """
    node = self.dom.parseString('<foo><bar>baz</bar><bar>quux</bar></foo>')
    self.assertEqual(domhelpers.getNodeText(node), 'bazquux')