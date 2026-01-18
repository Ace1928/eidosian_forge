import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def delete_message_batch(self, queue, messages):
    """
        Deletes a list of messages from a queue in a single request.

        :type queue: A :class:`boto.sqs.queue.Queue` object.
        :param queue: The Queue to which the messages will be written.

        :type messages: List of :class:`boto.sqs.message.Message` objects.
        :param messages: A list of message objects.
        """
    params = {}
    for i, msg in enumerate(messages):
        prefix = 'DeleteMessageBatchRequestEntry'
        p_name = '%s.%i.Id' % (prefix, i + 1)
        params[p_name] = msg.id
        p_name = '%s.%i.ReceiptHandle' % (prefix, i + 1)
        params[p_name] = msg.receipt_handle
    return self.get_object('DeleteMessageBatch', params, BatchResults, queue.id, verb='POST')