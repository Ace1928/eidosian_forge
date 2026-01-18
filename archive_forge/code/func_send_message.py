import time
from threading import Timer
from tests.unit import unittest
from boto.sqs.connection import SQSConnection
from boto.sqs.message import Message
from boto.sqs.message import MHMessage
from boto.exception import SQSError
def send_message():
    messages.append(queue.write(queue.new_message('this is a test message')))