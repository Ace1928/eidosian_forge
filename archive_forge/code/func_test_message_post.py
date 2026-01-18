import json
from unittest import mock
from zaqarclient import errors
from zaqarclient.queues import client
from zaqarclient.queues.v1 import iterator
from zaqarclient.queues.v1 import message
from zaqarclient.queues.v2 import subscription
from zaqarclient.tests.queues import base
from zaqarclient.transport import response
def test_message_post(self):
    messages = [{'ttl': 30, 'body': 'Post It!'}]
    result = {'resources': ['/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01'], 'partial': False}
    with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
        resp = response.Response(None, json.dumps(result))
        send_method.return_value = resp
        posted = self.queue.post(messages)
        self.assertEqual(result, posted)