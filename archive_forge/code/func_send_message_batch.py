import boto
from boto.connection import AWSQueryConnection
from boto.sqs.regioninfo import SQSRegionInfo
from boto.sqs.queue import Queue
from boto.sqs.message import Message
from boto.sqs.attributes import Attributes
from boto.sqs.batchresults import BatchResults
from boto.exception import SQSError, BotoServerError
def send_message_batch(self, queue, messages):
    """
        Delivers up to 10 messages to a queue in a single request.

        :type queue: A :class:`boto.sqs.queue.Queue` object.
        :param queue: The Queue to which the messages will be written.

        :type messages: List of lists.
        :param messages: A list of lists or tuples.  Each inner
            tuple represents a single message to be written
            and consists of and ID (string) that must be unique
            within the list of messages, the message body itself
            which can be a maximum of 64K in length, an
            integer which represents the delay time (in seconds)
            for the message (0-900) before the message will
            be delivered to the queue, and an optional dict of
            message attributes like those passed to ``send_message``
            above.

        """
    params = {}
    for i, msg in enumerate(messages):
        base = 'SendMessageBatchRequestEntry.%i' % (i + 1)
        params['%s.Id' % base] = msg[0]
        params['%s.MessageBody' % base] = msg[1]
        params['%s.DelaySeconds' % base] = msg[2]
        if len(msg) > 3:
            base += '.MessageAttribute'
            keys = sorted(msg[3].keys())
            for j, name in enumerate(keys):
                attribute = msg[3][name]
                p_name = '%s.%i.Name' % (base, j + 1)
                params[p_name] = name
                if 'data_type' in attribute:
                    p_name = '%s.%i.Value.DataType' % (base, j + 1)
                    params[p_name] = attribute['data_type']
                if 'string_value' in attribute:
                    p_name = '%s.%i.Value.StringValue' % (base, j + 1)
                    params[p_name] = attribute['string_value']
                if 'binary_value' in attribute:
                    p_name = '%s.%i.Value.BinaryValue' % (base, j + 1)
                    params[p_name] = attribute['binary_value']
                if 'string_list_value' in attribute:
                    p_name = '%s.%i.Value.StringListValue' % (base, j + 1)
                    params[p_name] = attribute['string_list_value']
                if 'binary_list_value' in attribute:
                    p_name = '%s.%i.Value.BinaryListValue' % (base, j + 1)
                    params[p_name] = attribute['binary_list_value']
    return self.get_object('SendMessageBatch', params, BatchResults, queue.id, verb='POST')