import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_claim_create_delete_functional(self):
    queue = self.client.queue('test_queue')
    queue._get_transport = mock.Mock(return_value=self.transport)
    messages = [{'ttl': 60, 'body': 'Post It 1!'}]
    queue.post(messages)
    cl = queue.claim(ttl=120, grace=120)
    claim_id = cl.id
    cl.delete()
    self.assertRaises(errors.ResourceNotFound, queue.claim, id=claim_id)