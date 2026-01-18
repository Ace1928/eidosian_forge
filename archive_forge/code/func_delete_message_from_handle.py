import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def delete_message_from_handle(self, queue, receipt_handle):
    """
        Delete a message from a queue, given a receipt handle.

        :type queue: A :class:`boto.sqs.queue.Queue` object
        :param queue: The Queue from which messages are read.

        :type receipt_handle: str
        :param receipt_handle: The receipt handle for the message

        :rtype: bool
        :return: True if successful, False otherwise.
        """
    params = {'ReceiptHandle': receipt_handle}
    return self.get_status('DeleteMessage', params, queue.id)