import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def put_queue_message(self, queue):
    m1 = Message()
    m1.set_body('This is a test message.')
    queue.write(m1)