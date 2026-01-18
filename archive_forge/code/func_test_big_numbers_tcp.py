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
def test_big_numbers_tcp():
    """Test big numbers with TCP client."""
    cl = _tcp_client()
    _test_big_numbers(cl, 'tcp')