import asyncio
import asyncio.events
import collections
import contextlib
import gc
import logging
import os
import pprint
import re
import select
import socket
import ssl
import sys
import tempfile
import threading
import time
import unittest
import uvloop
def has_IPv6():
    server_sock = socket.socket(socket.AF_INET6)
    with server_sock:
        try:
            server_sock.bind(('::1', 0))
        except OSError:
            return False
        else:
            return True