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
def test_ipv4_resolution_tcp():
    cl = _tcp_client(addr='localhost')
    _test_resolution(cl, 'tcp', ('127.0.0.1', 8125))