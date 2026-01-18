import json
import time
from unittest import mock
from zaqarclient.queues.v1 import claim
from zaqarclient.tests.queues import base
from zaqarclient.transport import errors
from zaqarclient.transport import response
def test_claim_get_by_id(self):
    result = {'href': '/v1/queues/fizbit/messages/50b68a50d6cb01?claim_id=4524', 'age': 790, 'ttl': 800, 'messages': [{'href': '/v1/queues/fizbit/messages/50b68a50d6f5b8c8a7c62b01', 'ttl': 800, 'age': 790, 'body': {'event': 'ActivateAccount', 'mode': 'active'}}]}
    with mock.patch.object(self.transport, 'send', autospec=True) as send_method:
        resp = response.Response(None, json.dumps(result))
        send_method.return_value = resp
        cl = self.queue.claim(id='5245432')
        num_tested = 0
        for num, msg in enumerate(cl):
            num_tested += 1
            self.assertEqual(result['messages'][num]['href'], msg.href)
        self.assertEqual(len(result['messages']), num_tested)