import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_event_creation_time_with_millis(self):
    millis_xml = SAMPLE_XML.replace(b'<CreationTime>2013-01-10T05:04:56Z</CreationTime>', b'<CreationTime>2013-01-10T05:04:56.102342Z</CreationTime>')
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.Stack)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(millis_xml, h)
    creation_time = rs[0].creation_time
    self.assertEqual(creation_time, datetime.datetime(2013, 1, 10, 5, 4, 56, 102342))