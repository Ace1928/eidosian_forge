import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def delete_message(self, queue, message):
    """
        Delete a message from a queue.

        :type queue: A :class:`boto.sqs.queue.Queue` object
        :param queue: The Queue from which messages are read.

        :type message: A :class:`boto.sqs.message.Message` object
        :param message: The Message to be deleted

        :rtype: bool
        :return: True if successful, False otherwise.
        """
    params = {'ReceiptHandle': message.receipt_handle}
    return self.get_status('DeleteMessage', params, queue.id)