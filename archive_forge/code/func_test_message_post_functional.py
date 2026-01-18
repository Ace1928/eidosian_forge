import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_post_functional(self):
    messages = [{'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}, {'ttl': 60, 'body': 'Post It!'}]
    queue = self.client.queue('nonono')
    queue._get_transport = mock.Mock(return_value=self.transport)
    result = queue.post(messages)
    self.assertIn('resources', result)
    self.assertEqual(3, len(result['resources']))