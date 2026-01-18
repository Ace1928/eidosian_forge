import os
import tempfile
from xml import sax
from six import StringIO
from boto import handler
from boto.exception import GSResponseError
from boto.gs.acl import ACL
from tests.integration.gs.testcase import GSTestCase
def testBucketConditionalSetXmlAcl(self):
    b = self._MakeVersionedBucket()
    k = b.new_key('foo')
    s1 = 'test1'
    k.set_contents_from_string(s1)
    g1 = k.generation
    mg1 = k.metageneration
    self.assertEqual(str(mg1), '1')
    acl_xml = '<ACCESSControlList><EntrIes><Entry>' + '<Scope type="AllUsers"></Scope><Permission>READ</Permission>' + '</Entry></EntrIes></ACCESSControlList>'
    acl = ACL()
    h = handler.XmlHandler(acl, b)
    sax.parseString(acl_xml, h)
    acl = acl.to_xml()
    b.set_xml_acl(acl, key_name='foo')
    k = b.get_key('foo')
    g2 = k.generation
    mg2 = k.metageneration
    self.assertEqual(g2, g1)
    self.assertGreater(mg2, mg1)
    with self.assertRaisesRegexp(ValueError, 'Received if_metageneration argument with no if_generation argument'):
        b.set_xml_acl(acl, key_name='foo', if_metageneration=123)
    with self.assertRaisesRegexp(GSResponseError, VERSION_MISMATCH):
        b.set_xml_acl(acl, key_name='foo', if_generation=int(g2) + 1)
    with self.assertRaisesRegexp(GSResponseError, VERSION_MISMATCH):
        b.set_xml_acl(acl, key_name='foo', if_generation=g2, if_metageneration=int(mg2) + 1)
    b.set_xml_acl(acl, key_name='foo', if_generation=g2)
    k = b.get_key('foo')
    g3 = k.generation
    mg3 = k.metageneration
    self.assertEqual(g3, g2)
    self.assertGreater(mg3, mg2)
    b.set_xml_acl(acl, key_name='foo', if_generation=g3, if_metageneration=mg3)