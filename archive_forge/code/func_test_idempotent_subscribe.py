import hashlib
import time
from tests.unit import unittest
from boto.compat import json
from boto.sqs.connection import SQSConnection
from boto.sns.connection import SNSConnection
def test_idempotent_subscribe(self):
    now = time.time()
    topic_name = queue_name = 'test_idempotent_subscribe%d' % now
    timeout = 60
    queue = self.sqsc.create_queue(queue_name, timeout)
    self.addCleanup(self.sqsc.delete_queue, queue, True)
    initial_statements = self.get_policy_statements(queue)
    queue_arn = queue.arn
    topic = self.snsc.create_topic(topic_name)
    topic_arn = topic['CreateTopicResponse']['CreateTopicResult']['TopicArn']
    self.addCleanup(self.snsc.delete_topic, topic_arn)
    resp = self.snsc.subscribe_sqs_queue(topic_arn, queue)
    time.sleep(3)
    first_subscribe_statements = self.get_policy_statements(queue)
    self.assertEqual(len(first_subscribe_statements), len(initial_statements) + 1)
    resp2 = self.snsc.subscribe_sqs_queue(topic_arn, queue)
    time.sleep(3)
    second_subscribe_statements = self.get_policy_statements(queue)
    self.assertEqual(len(second_subscribe_statements), len(first_subscribe_statements))