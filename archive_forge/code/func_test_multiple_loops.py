import asyncio
import json
import os
import sys
from concurrent.futures import CancelledError
from multiprocessing import Process
import pytest
from pytest import mark
import zmq
import zmq.asyncio as zaio
def test_multiple_loops(push_pull):
    a, b = push_pull

    async def test():
        await a.send(b'buf')
        msg = await b.recv()
        assert msg == b'buf'
    for i in range(3):
        loop = asyncio.new_event_loop()
        loop.run_until_complete(asyncio.wait_for(test(), timeout=10))
        loop.close()