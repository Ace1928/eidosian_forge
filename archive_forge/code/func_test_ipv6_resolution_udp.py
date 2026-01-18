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
def test_ipv6_resolution_udp():
    raise SkipTest('IPv6 resolution is broken on Travis')
    cl = _udp_client(addr='localhost', ipv6=True)
    _test_resolution(cl, 'udp', ('::1', 8125, 0, 0))