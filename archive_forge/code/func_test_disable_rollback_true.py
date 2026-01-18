import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_disable_rollback_true(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
    h = boto.handler.XmlHandler(rs, None)
    sample_xml_upper = SAMPLE_XML.replace(b'false', b'true')
    xml.sax.parseString(sample_xml_upper, h)
    disable_rollback = rs[0].disable_rollback
    self.assertTrue(disable_rollback)