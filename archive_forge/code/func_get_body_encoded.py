import base64
import boto
from boto.compat import StringIO
from boto.compat import six
from boto.sqs.attributes import Attributes
from boto.sqs.messageattributes import MessageAttributes
from boto.exception import SQSDecodeError
def get_body_encoded(self):
    """
        This method is really a semi-private method used by the Queue.write
        method when writing the contents of the message to SQS.
        You probably shouldn't need to call this method in the normal course of events.
        """
    return self.encode(self.get_body())