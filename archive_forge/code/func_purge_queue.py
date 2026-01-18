import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def purge_queue(self, queue):
    """
        Purge all messages in an SQS Queue.

        :type queue: A Queue object
        :param queue: The SQS queue to be purged

        :rtype: bool
        :return: True if the command succeeded, False otherwise
        """
    return self.get_status('PurgeQueue', None, queue.id)