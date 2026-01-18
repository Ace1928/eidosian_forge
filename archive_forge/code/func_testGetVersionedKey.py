from xml import sax
from boto import handler
from boto.gs import acl
from tests.integration.gs.testcase import GSTestCase
def testGetVersionedKey(self):
    b = self._MakeVersionedBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    k.set_contents_from_string(s1)
    k = b.get_key('foo')
    g1 = k.generation
    o1 = k.get_contents_as_string()
    self.assertEqual(o1, s1)
    s2 = 'test2'
    k.set_contents_from_string(s2)
    k = b.get_key('foo')
    g2 = k.generation
    self.assertNotEqual(g2, g1)
    o2 = k.get_contents_as_string()
    self.assertEqual(o2, s2)
    k = b.get_key('foo', generation=g1)
    self.assertEqual(k.get_contents_as_string(), s1)
    k = b.get_key('foo', generation=g2)
    self.assertEqual(k.get_contents_as_string(), s2)