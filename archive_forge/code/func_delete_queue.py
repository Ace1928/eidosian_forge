import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def delete_queue(self, queue, force_deletion=False):
    """
        Delete an SQS Queue.

        :type queue: A Queue object
        :param queue: The SQS queue to be deleted

        :type force_deletion: Boolean
        :param force_deletion: A deprecated parameter that is no longer used by
            SQS's API.

        :rtype: bool
        :return: True if the command succeeded, False otherwise
        """
    return self.get_status('DeleteQueue', None, queue.id)