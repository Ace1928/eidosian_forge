import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def create_temp_queue(self, conn):
    current_timestamp = int(time.time())
    queue_name = 'test%d' % int(time.time())
    test = conn.create_queue(queue_name)
    self.addCleanup(conn.delete_queue, test)
    return test