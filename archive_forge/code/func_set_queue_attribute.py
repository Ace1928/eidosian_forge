import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def set_queue_attribute(self, queue, attribute, value):
    """
        Set a new value for an attribute of a Queue.

        :type queue: A Queue object
        :param queue: The SQS queue to get attributes for

        :type attribute: String
        :param attribute: The name of the attribute you want to set.

        :param value: The new value for the attribute must be:

            * For `DelaySeconds` the value must be an integer number of
            seconds from 0 to 900 (15 minutes).
                >>> connection.set_queue_attribute(queue, 'DelaySeconds', 900)

            * For `MaximumMessageSize` the value must be an integer number of
            bytes from 1024 (1 KiB) to 262144 (256 KiB).
                >>> connection.set_queue_attribute(queue, 'MaximumMessageSize', 262144)

            * For `MessageRetentionPeriod` the value must be an integer number of
            seconds from 60 (1 minute) to 1209600 (14 days).
                >>> connection.set_queue_attribute(queue, 'MessageRetentionPeriod', 1209600)

            * For `Policy` the value must be an string that contains JSON formatted
            parameters and values.
                >>> connection.set_queue_attribute(queue, 'Policy', json.dumps({
                ...     'Version': '2008-10-17',
                ...     'Id': '/123456789012/testQueue/SQSDefaultPolicy',
                ...     'Statement': [
                ...        {
                ...            'Sid': 'Queue1ReceiveMessage',
                ...            'Effect': 'Allow',
                ...            'Principal': {
                ...                'AWS': '*'
                ...            },
                ...            'Action': 'SQS:ReceiveMessage',
                ...            'Resource': 'arn:aws:aws:sqs:us-east-1:123456789012:testQueue'
                ...        }
                ...    ]
                ... }))

            * For `ReceiveMessageWaitTimeSeconds` the value must be an integer number of
            seconds from 0 to 20.
                >>> connection.set_queue_attribute(queue, 'ReceiveMessageWaitTimeSeconds', 20)

            * For `VisibilityTimeout` the value must be an integer number of
            seconds from 0 to 43200 (12 hours).
                >>> connection.set_queue_attribute(queue, 'VisibilityTimeout', 43200)

            * For `RedrivePolicy` the value must be an string that contains JSON formatted
            parameters and values. You can set maxReceiveCount to a value between 1 and 1000.
            The deadLetterTargetArn value is the Amazon Resource Name (ARN) of the queue that
            will receive the dead letter messages.
                >>> connection.set_queue_attribute(queue, 'RedrivePolicy', json.dumps({
                ...    'maxReceiveCount': 5,
                ...    'deadLetterTargetArn': "arn:aws:aws:sqs:us-east-1:123456789012:testDeadLetterQueue"
                ... }))
        """
    params = {'Attribute.Name': attribute, 'Attribute.Value': value}
    return self.get_status('SetQueueAttributes', params, queue.id)