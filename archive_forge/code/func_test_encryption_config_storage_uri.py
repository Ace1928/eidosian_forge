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
def test_encryption_config_storage_uri(self):
    """Test setting and getting of EncryptionConfig with storage_uri."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    uri = storage_uri('gs://' + bucket_name)
    encryption_config = uri.get_encryption_config()
    self.assertIsNone(encryption_config.default_kms_key_name)
    uri.set_encryption_config()