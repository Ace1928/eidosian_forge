import base64
import boto
from boto.compat import StringIO
from boto.compat import six
from boto.sqs.attributes import Attributes
from boto.sqs.messageattributes import MessageAttributes
from boto.exception import SQSDecodeError
def endNode(self, connection):
    self.set_body(self.decode(self.get_body()))