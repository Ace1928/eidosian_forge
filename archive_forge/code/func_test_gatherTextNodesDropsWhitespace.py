from importlib import reload
from typing import Any, Optional
from xml.dom import minidom
from twisted.trial.unittest import TestCase
from twisted.web import domhelpers, microdom
def test_gatherTextNodesDropsWhitespace(self):
    """
        Microdom discards whitespace-only text nodes, so L{gatherTextNodes}
        returns only the text from nodes which had non-whitespace characters.
        """
    doc4_xml = '<html>\n  <head>\n  </head>\n  <body>\n    stuff\n  </body>\n</html>\n'
    doc4 = self.dom.parseString(doc4_xml)
    actual = domhelpers.gatherTextNodes(doc4)
    expected = '\n    stuff\n  '
    self.assertEqual(actual, expected)
    actual = domhelpers.gatherTextNodes(doc4.documentElement)
    self.assertEqual(actual, expected)