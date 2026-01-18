from tests.unit import unittest
from boto.sqs.message import MHMessage
from boto.sqs.message import RawMessage
from boto.sqs.message import Message
from boto.sqs.bigmessage import BigMessage
from boto.exception import SQSDecodeError
from nose.plugins.attrib import attr
@attr(sqs=True)
def test_message_id_available(self):
    import xml.sax
    from boto.resultset import ResultSet
    from boto.handler import XmlHandler
    sample_value = 'abcdef'
    body = '<?xml version="1.0"?>\n            <ReceiveMessageResponse>\n              <ReceiveMessageResult>\n                <Message>\n                  <Body>%s</Body>\n                  <ReceiptHandle>%s</ReceiptHandle>\n                  <MessageId>%s</MessageId>\n                </Message>\n              </ReceiveMessageResult>\n            </ReceiveMessageResponse>' % tuple([sample_value] * 3)
    rs = ResultSet([('Message', DecodeExceptionRaisingMessage)])
    h = XmlHandler(rs, None)
    with self.assertRaises(SQSDecodeError) as context:
        xml.sax.parseString(body.encode('utf-8'), h)
    message = context.exception.message
    self.assertEquals(message.id, sample_value)
    self.assertEquals(message.receipt_handle, sample_value)