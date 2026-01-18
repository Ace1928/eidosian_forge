import base64
import boto
from boto.compat import StringIO
from boto.compat import six
from boto.sqs.attributes import Attributes
from boto.sqs.messageattributes import MessageAttributes
from boto.exception import SQSDecodeError
def change_visibility(self, visibility_timeout):
    if self.queue:
        self.queue.connection.change_message_visibility(self.queue, self.receipt_handle, visibility_timeout)