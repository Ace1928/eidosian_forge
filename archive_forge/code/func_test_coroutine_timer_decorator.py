import asyncio
import functools
import random
import re
import socket
from datetime import timedelta
from unittest import SkipTest, mock
from statsd import StatsClient
from statsd import TCPStatsClient
from statsd import UnixSocketStatsClient
def test_coroutine_timer_decorator():
    event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(event_loop)
    already = False
    cl = _udp_client()

    def _send(*_):
        nonlocal already
        assert already is True, '_send called before coroutine completed'
    cl._send = _send

    @cl.timer('bar')
    async def inner():
        nonlocal already
        await asyncio.sleep(0)
        already = True
        return None
    event_loop.run_until_complete(inner())
    event_loop.close()