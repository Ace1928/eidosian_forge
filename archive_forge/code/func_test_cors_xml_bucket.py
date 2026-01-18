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
def test_cors_xml_bucket(self):
    """Test setting and getting of CORS XML documents on Bucket."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    bucket = self._GetConnection().get_bucket(bucket_name)
    cors = re.sub('\\s', '', bucket.get_cors().to_xml())
    self.assertEqual(cors, CORS_EMPTY)
    bucket.set_cors(CORS_DOC)
    cors = re.sub('\\s', '', bucket.get_cors().to_xml())
    self.assertEqual(cors, CORS_DOC)