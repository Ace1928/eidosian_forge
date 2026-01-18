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
def test_ipv4_resolution_udp():
    cl = _udp_client(addr='localhost')
    _test_resolution(cl, 'udp', ('127.0.0.1', 8125))