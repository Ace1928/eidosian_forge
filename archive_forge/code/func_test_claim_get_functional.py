import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_claim_get_functional(self):
    queue = self.client.queue('test_queue')
    queue._get_transport = mock.Mock(return_value=self.transport)
    res = queue.claim(ttl=100, grace=100)
    claim_id = res.id
    cl = queue.claim(id=claim_id)
    self.assertEqual(claim_id, cl.id)