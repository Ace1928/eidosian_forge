import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_get_many_functional(self):
    queue = self.client.queue('test_queue')
    self.addCleanup(queue.delete)
    queue._get_transport = mock.Mock(return_value=self.transport)
    messages = [{'ttl': 60, 'body': 'Post It 1!'}, {'ttl': 60, 'body': 'Post It 2!'}, {'ttl': 60, 'body': 'Post It 3!'}]
    res = queue.post(messages)['resources']
    msgs_id = [ref.split('/')[-1] for ref in res]
    messages = queue.messages(*msgs_id)
    self.assertIsInstance(messages, iterator._Iterator)
    messages = list(messages)
    length = len(messages)
    if length == 3:
        bodies = set((message.body for message in messages))
        self.assertEqual(set(['Post It 1!', 'Post It 2!', 'Post It 3!']), bodies)
    elif length == 1:
        pass
    else:
        self.fail("Wrong number of messages: '%d'" % length)