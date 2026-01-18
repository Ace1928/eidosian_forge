from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_namespaceReuse(self):
    """
        Test that reuse of namespaces does affect an element's serialization.

        When one element uses a prefix for a certain namespace, this is
        stored in the C{localPrefixes} attribute of the element. We want
        to make sure that elements created after such use, won't have this
        prefix end up in their C{localPrefixes} attribute, too.
        """
    xml = b"<root>\n                    <foo:child1 xmlns:foo='testns'/>\n                    <child2 xmlns='testns'/>\n                  </root>"
    self.stream.parse(xml)
    self.assertEqual('child1', self.elements[0].name)
    self.assertEqual('testns', self.elements[0].uri)
    self.assertEqual('', self.elements[0].defaultUri)
    self.assertEqual({'foo': 'testns'}, self.elements[0].localPrefixes)
    self.assertEqual('child2', self.elements[1].name)
    self.assertEqual('testns', self.elements[1].uri)
    self.assertEqual('testns', self.elements[1].defaultUri)
    self.assertEqual({}, self.elements[1].localPrefixes)