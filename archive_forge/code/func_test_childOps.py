from zope.interface.verify import verifyObject
from twisted.python.reflect import requireModule
from twisted.trial import unittest
from twisted.words.xish import domish
def test_childOps(self):
    """
        Basic L{domish.Element} child tests.
        """
    e = domish.Element(('testns', 'foo'))
    e.addContent('somecontent')
    b2 = e.addElement(('testns2', 'bar2'))
    e['attrib1'] = 'value1'
    e['testns2', 'attrib2'] = 'value2'
    e.addElement('bar')
    e.addElement('bar')
    e.addContent('abc')
    e.addContent('123')
    self.assertEqual(e.children[-1], 'abc123')
    self.assertEqual(e.bar2, b2)
    e.bar2.addContent('subcontent')
    e.bar2['bar2value'] = 'somevalue'
    self.assertEqual(e.children[1], e.bar2)
    self.assertEqual(e.children[2], e.bar)
    self.assertEqual(e['attrib1'], 'value1')
    del e['attrib1']
    self.assertEqual(e.hasAttribute('attrib1'), 0)
    self.assertEqual(e.hasAttribute('attrib2'), 0)
    self.assertEqual(e['testns2', 'attrib2'], 'value2')