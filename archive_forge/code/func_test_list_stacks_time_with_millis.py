import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_list_stacks_time_with_millis(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackSummary)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(LIST_STACKS_XML, h)
    timestamp_1 = rs[0].creation_time
    self.assertEqual(timestamp_1, datetime.datetime(2011, 5, 23, 15, 47, 44))
    timestamp_2 = rs[1].creation_time
    self.assertEqual(timestamp_2, datetime.datetime(2011, 3, 5, 19, 57, 58, 161616))
    timestamp_3 = rs[1].deletion_time
    self.assertEqual(timestamp_3, datetime.datetime(2011, 3, 10, 16, 20, 51, 575757))