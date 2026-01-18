import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_custom_serialize_error(self):

    async def test():
        a, b = self.create_bound_pair(zmq.DEALER, zmq.ROUTER)
        msg = {'content': {'a': 5, 'b': 'bee'}}
        with pytest.raises(TypeError):
            await a.send_serialized(json, json.dumps)
        await a.send(b'not json')
        with pytest.raises(TypeError):
            await b.recv_serialized(json.loads)
    self.loop.run_sync(test)