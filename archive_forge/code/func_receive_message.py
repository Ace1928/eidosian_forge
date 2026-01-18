import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def receive_message(self, queue, number_messages=1, visibility_timeout=None, attributes=None, wait_time_seconds=None, message_attributes=None):
    """
        Read messages from an SQS Queue.

        :type queue: A Queue object
        :param queue: The Queue from which messages are read.

        :type number_messages: int
        :param number_messages: The maximum number of messages to read
                                (default=1)

        :type visibility_timeout: int
        :param visibility_timeout: The number of seconds the message should
            remain invisible to other queue readers
            (default=None which uses the Queues default)

        :type attributes: str
        :param attributes: The name of additional attribute to return
            with response or All if you want all attributes.  The
            default is to return no additional attributes.  Valid
            values:
            * All
            * SenderId
            * SentTimestamp
            * ApproximateReceiveCount
            * ApproximateFirstReceiveTimestamp

        :type wait_time_seconds: int
        :param wait_time_seconds: The duration (in seconds) for which the call
            will wait for a message to arrive in the queue before returning.
            If a message is available, the call will return sooner than
            wait_time_seconds.

        :type message_attributes: list
        :param message_attributes: The name(s) of additional message
            attributes to return. The default is to return no additional
            message attributes. Use ``['All']`` or ``['.*']`` to return all.

        :rtype: list
        :return: A list of :class:`boto.sqs.message.Message` objects.

        """
    params = {'MaxNumberOfMessages': number_messages}
    if visibility_timeout is not None:
        params['VisibilityTimeout'] = visibility_timeout
    if attributes is not None:
        self.build_list_params(params, attributes, 'AttributeName')
    if wait_time_seconds is not None:
        params['WaitTimeSeconds'] = wait_time_seconds
    if message_attributes is not None:
        self.build_list_params(params, message_attributes, 'MessageAttributeName')
    return self.get_list('ReceiveMessage', params, [('Message', queue.message_class)], queue.id, queue)