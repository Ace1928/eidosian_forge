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
@mock.patch.object(random, 'random')
def test_gauge_absolute_negative_rate_tcp(mock_random):
    """TCPStatsClient.gauge works with absolute negative value and rate."""
    cl = _tcp_client()
    _test_gauge_absolute_negative_rate(cl, 'tcp', mock_random)