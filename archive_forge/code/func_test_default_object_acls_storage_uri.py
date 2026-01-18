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
def test_default_object_acls_storage_uri(self):
    """Test default object acls using storage_uri."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    uri = storage_uri('gs://' + bucket_name)
    acl = uri.get_def_acl()
    self.assertIsNotNone(re.search(PROJECT_PRIVATE_RE, acl.to_xml()), 'PROJECT_PRIVATE_RE not found in ACL XML:\n' + acl.to_xml())
    uri.set_def_acl('public-read')
    acl = uri.get_def_acl()
    public_read_acl = acl
    self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
    uri.set_def_acl('private')
    acl = uri.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')
    uri.set_def_acl(public_read_acl)
    acl = uri.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList><Entries><Entry><Scope type="AllUsers"></Scope><Permission>READ</Permission></Entry></Entries></AccessControlList>')
    uri.set_def_acl('private')
    acl = uri.get_def_acl()
    self.assertEqual(acl.to_xml(), '<AccessControlList></AccessControlList>')