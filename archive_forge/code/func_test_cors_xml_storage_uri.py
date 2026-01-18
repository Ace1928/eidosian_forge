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
def test_cors_xml_storage_uri(self):
    """Test setting and getting of CORS XML documents with storage_uri."""
    bucket = self._MakeBucket()
    bucket_name = bucket.name
    uri = storage_uri('gs://' + bucket_name)
    cors = re.sub('\\s', '', uri.get_cors().to_xml())
    self.assertEqual(cors, CORS_EMPTY)
    cors_obj = Cors()
    h = handler.XmlHandler(cors_obj, None)
    xml.sax.parseString(CORS_DOC, h)
    uri.set_cors(cors_obj)
    cors = re.sub('\\s', '', uri.get_cors().to_xml())
    self.assertEqual(cors, CORS_DOC)