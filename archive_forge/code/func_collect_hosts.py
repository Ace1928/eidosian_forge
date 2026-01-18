import asyncio
import collections
import base64
import functools
import hashlib
import hmac
import logging
import random
import socket
import struct
import sys
import time
import traceback
import uuid
import warnings
import weakref
import async_timeout
import aiokafka.errors as Errors
from aiokafka.abc import AbstractTokenProvider
from aiokafka.protocol.api import RequestHeader
from aiokafka.protocol.admin import (
from aiokafka.protocol.commit import (
from aiokafka.util import create_future, create_task, get_running_loop, wait_for
def collect_hosts(hosts, randomize=True):
    """
    Collects a comma-separated set of hosts (host:port) and optionally
    randomize the returned list.
    """
    if isinstance(hosts, str):
        hosts = hosts.strip().split(',')
    result = []
    afi = socket.AF_INET
    for host_port in hosts:
        host, port, afi = get_ip_port_afi(host_port)
        if port < 0:
            port = DEFAULT_KAFKA_PORT
        result.append((host, port, afi))
    if randomize:
        random.shuffle(result)
    return result