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
def test_encryption_config_bucket(self):
    """Test setting and getting of EncryptionConfig on gs Bucket objects."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    encryption_config = bucket.get_encryption_config()
    self.assertIsNone(encryption_config.default_kms_key_name)
    xmldoc = bucket._construct_encryption_config_xml(default_kms_key_name='dummykey')
    self.assertEqual(xmldoc, ENCRYPTION_CONFIG_WITH_KEY % 'dummykey')
    bucket.set_encryption_config()