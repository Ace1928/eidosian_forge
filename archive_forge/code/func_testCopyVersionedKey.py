from xml import sax
from boto import handler
from boto.gs import acl
from tests.integration.gs.testcase import GSTestCase
def testCopyVersionedKey(self):
    b = self._MakeVersionedBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    k.set_contents_from_string(s1)
    k = b.get_key('foo')
    g1 = k.generation
    s2 = 'test2'
    k.set_contents_from_string(s2)
    b2 = self._MakeVersionedBucket()
    b2.copy_key('foo2', b.name, 'foo', src_generation=g1)
    k2 = b2.get_key('foo2')
    s3 = k2.get_contents_as_string()
    self.assertEqual(s3, s1)