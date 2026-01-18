import os
import re
import urllib
import xml.sax
from six import StringIO
from boto import handler
from boto import storage_uri
from boto.gs.acl import ACL
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from tests.integration.gs.testcase import GSTestCase
def test_get_all_keys(self):
    """Tests get_all_keys."""
    phony_mimetype = 'application/x-boto-test'
    headers = {'Content-Type': phony_mimetype}
    tmpdir = self._MakeTempDir()
    fpath = os.path.join(tmpdir, 'foobar1')
    fpath2 = os.path.join(tmpdir, 'foobar')
    with open(fpath2, 'w') as f:
        f.write('test-data')
    bucket = self._MakeBucket()
    k = bucket.new_key('foobar')
    s1 = 'test-contents'
    s2 = 'test-contents2'
    k.name = 'foo/bar'
    k.set_contents_from_string(s1, headers)
    k.name = 'foo/bas'
    k.set_contents_from_filename(fpath2)
    k.name = 'foo/bat'
    k.set_contents_from_string(s1)
    k.name = 'fie/bar'
    k.set_contents_from_string(s1)
    k.name = 'fie/bas'
    k.set_contents_from_string(s1)
    k.name = 'fie/bat'
    k.set_contents_from_string(s1)
    md5 = k.md5
    k.set_contents_from_string(s2)
    self.assertNotEqual(k.md5, md5)
    fp2 = open(fpath2, 'rb')
    k.md5 = None
    k.base64md5 = None
    k.set_contents_from_stream(fp2)
    fp = open(fpath, 'wb')
    k.get_contents_to_file(fp)
    fp.close()
    fp2.seek(0, 0)
    fp = open(fpath, 'rb')
    self.assertEqual(fp2.read(), fp.read())
    fp.close()
    fp2.close()
    all = bucket.get_all_keys()
    self.assertEqual(len(all), 6)
    rs = bucket.get_all_keys(prefix='foo')
    self.assertEqual(len(rs), 3)
    rs = bucket.get_all_keys(prefix='', delimiter='/')
    self.assertEqual(len(rs), 2)
    rs = bucket.get_all_keys(maxkeys=5)
    self.assertEqual(len(rs), 5)