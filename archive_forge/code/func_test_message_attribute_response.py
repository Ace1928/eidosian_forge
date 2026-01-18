from tests.unit import AWSMockServiceTestCase, MockServiceWithConfigTestCase
from tests.compat import mock
from boto.sqs.connection import SQSConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.message import RawMessage
from boto.sqs.queue import Queue
from boto.connection import AWSQueryConnection
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_message_attribute_response(self):
    self.set_http_response(status_code=200)
    queue = Queue(url='http://sqs.us-east-1.amazonaws.com/123456789012/testQueue/', message_class=RawMessage)
    message = self.service_connection.receive_message(queue)[0]
    self.assertEqual(message.get_body(), 'This is a test')
    self.assertEqual(message.id, '7049431b-e5f6-430b-93c4-ded53864d02b')
    self.assertEqual(message.md5, 'ce114e4501d2f4e2dcea3e17b546f339')
    self.assertEqual(message.md5_message_attributes, '324758f82d026ac6ec5b31a3b192d1e3')
    mattributes = message.message_attributes
    self.assertEqual(len(mattributes.keys()), 2)
    self.assertEqual(mattributes['Count']['data_type'], 'Number')
    self.assertEqual(mattributes['Foo']['string_value'], 'Bar')