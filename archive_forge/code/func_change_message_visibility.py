import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def change_message_visibility(self, queue, receipt_handle, visibility_timeout):
    """
        Extends the read lock timeout for the specified message from
        the specified queue to the specified value.

        :type queue: A :class:`boto.sqs.queue.Queue` object
        :param queue: The Queue from which messages are read.

        :type receipt_handle: str
        :param receipt_handle: The receipt handle associated with the message
                               whose visibility timeout will be changed.

        :type visibility_timeout: int
        :param visibility_timeout: The new value of the message's visibility
                                   timeout in seconds.
        """
    params = {'ReceiptHandle': receipt_handle, 'VisibilityTimeout': visibility_timeout}
    return self.get_status('ChangeMessageVisibility', params, queue.id)