import json
import os
import sys
from datetime import timedelta
import pytest
from tornado.ioloop import IOLoop
import zmq
from zmq.eventloop import future
from zmq.tests import BaseZMQTestCase
def test_close_all_fds(self):
    s = self.socket(zmq.PUB)

    async def attach():
        s._get_loop()
    self.loop.run_sync(attach)
    self.loop.close(all_fds=True)
    self.loop = None
    assert s.closed