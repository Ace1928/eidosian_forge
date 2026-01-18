from xml import sax
from boto import handler
from boto.gs import acl
from tests.integration.gs.testcase import GSTestCase
def testDeleteVersionedKey(self):
    b = self._MakeVersionedBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    k.set_contents_from_string(s1)
    k = b.get_key('foo')
    g1 = k.generation
    s2 = 'test2'
    k.set_contents_from_string(s2)
    k = b.get_key('foo')
    g2 = k.generation
    versions = list(b.list_versions())
    self.assertEqual(len(versions), 2)
    self.assertEqual(versions[0].name, 'foo')
    self.assertEqual(versions[1].name, 'foo')
    generations = [k.generation for k in versions]
    self.assertIn(g1, generations)
    self.assertIn(g2, generations)
    b.delete_key('foo', generation=None)
    self.assertIsNone(b.get_key('foo'))
    versions = list(b.list_versions())
    self.assertEqual(len(versions), 2)
    self.assertEqual(versions[0].name, 'foo')
    self.assertEqual(versions[1].name, 'foo')
    generations = [k.generation for k in versions]
    self.assertIn(g1, generations)
    self.assertIn(g2, generations)
    b.delete_key('foo', generation=g2)
    versions = list(b.list_versions())
    self.assertEqual(len(versions), 1)
    self.assertEqual(versions[0].name, 'foo')
    self.assertEqual(versions[0].generation, g1)
    b.delete_key('foo', generation=g1)
    versions = list(b.list_versions())
    self.assertEqual(len(versions), 0)