import binascii
import re
from six import StringIO
from boto import storage_uri
from boto.exception import BotoClientError
from boto.gs.acl import SupportedPermissions as perms
from tests.integration.gs.testcase import GSTestCase
def testHasVersion(self):
    uri = storage_uri('gs://bucket/obj')
    self.assertFalse(uri.has_version())
    uri.version_id = 'versionid'
    self.assertTrue(uri.has_version())
    uri = storage_uri('gs://bucket/obj')
    uri.generation = 12345
    self.assertTrue(uri.has_version())
    uri.generation = None
    self.assertFalse(uri.has_version())
    uri = storage_uri('gs://bucket/obj')
    uri.generation = 0
    self.assertTrue(uri.has_version())