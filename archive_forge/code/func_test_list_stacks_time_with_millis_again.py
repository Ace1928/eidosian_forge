import datetime
import xml.sax
import unittest
import boto.handler
import boto.resultset
import boto.cloudformation
def test_list_stacks_time_with_millis_again(self):
    rs = boto.resultset.ResultSet([('member', boto.cloudformation.stack.StackResourceSummary)])
    h = boto.handler.XmlHandler(rs, None)
    xml.sax.parseString(LIST_STACK_RESOURCES_XML, h)
    timestamp_1 = rs[0].last_updated_time
    self.assertEqual(timestamp_1, datetime.datetime(2011, 6, 21, 20, 15, 58))
    timestamp_2 = rs[1].last_updated_time
    self.assertEqual(timestamp_2, datetime.datetime(2011, 6, 21, 20, 25, 57, 875643))