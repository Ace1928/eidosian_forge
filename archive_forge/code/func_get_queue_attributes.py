import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def get_queue_attributes(self, queue, attribute='All'):
    """
        Gets one or all attributes of a Queue

        :type queue: A Queue object
        :param queue: The SQS queue to get attributes for

        :type attribute: str
        :param attribute: The specific attribute requested.  If not
            supplied, the default is to return all attributes.  Valid
            attributes are:

            * All
            * ApproximateNumberOfMessages
            * ApproximateNumberOfMessagesNotVisible
            * VisibilityTimeout
            * CreatedTimestamp
            * LastModifiedTimestamp
            * Policy
            * MaximumMessageSize
            * MessageRetentionPeriod
            * QueueArn
            * ApproximateNumberOfMessagesDelayed
            * DelaySeconds
            * ReceiveMessageWaitTimeSeconds
            * RedrivePolicy

        :rtype: :class:`boto.sqs.attributes.Attributes`
        :return: An Attributes object containing request value(s).
        """
    params = {'AttributeName': attribute}
    return self.get_object('GetQueueAttributes', params, Attributes, queue.id)