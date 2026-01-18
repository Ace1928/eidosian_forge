import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
def testPropertiesUpdated(self):
    b = self._MakeBucket()
    bucket_uri = storage_uri('gs://%s' % b.name)
    key_uri = bucket_uri.clone_replace_name('obj')
    key_uri.set_contents_from_string('data1')
    self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
    k = b.get_key('obj')
    self.assertEqual(k.generation, key_uri.generation)
    self.assertEquals(k.get_contents_as_string(), 'data1')
    key_uri.set_contents_from_stream(StringIO.StringIO('data2'))
    self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
    self.assertGreater(key_uri.generation, k.generation)
    k = b.get_key('obj')
    self.assertEqual(k.generation, key_uri.generation)
    self.assertEquals(k.get_contents_as_string(), 'data2')
    key_uri.set_contents_from_file(StringIO.StringIO('data3'))
    self.assertRegexpMatches(str(key_uri.generation), '[0-9]+')
    self.assertGreater(key_uri.generation, k.generation)
    k = b.get_key('obj')
    self.assertEqual(k.generation, key_uri.generation)
    self.assertEquals(k.get_contents_as_string(), 'data3')