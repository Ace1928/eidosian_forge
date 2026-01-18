from __future__ import annotations
import json
from botocore.serialize import Serializer
from vine import transform
from kombu.asynchronous.aws.connection import AsyncAWSQueryConnection
from kombu.asynchronous.aws.ext import AWSRequest
from .ext import boto3
from .message import AsyncMessage
from .queue import AsyncQueue
def get_queue_url(self, queue):
    res = self.sqs_connection.get_queue_url(QueueName=queue)
    return res['QueueUrl']