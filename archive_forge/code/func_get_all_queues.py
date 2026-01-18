import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def get_all_queues(self, prefix=''):
    """
        Retrieves all queues.

        :keyword str prefix: Optionally, only return queues that start with
            this value.
        :rtype: list
        :returns: A list of :py:class:`boto.sqs.queue.Queue` instances.
        """
    params = {}
    if prefix:
        params['QueueNamePrefix'] = prefix
    return self.get_list('ListQueues', params, [('QueueUrl', Queue)])