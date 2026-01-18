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
@contextlib.contextmanager
def silence_long_exec_warning():

    class Filter(logging.Filter):

        def filter(self, record):
            return not (record.msg.startswith('Executing') and record.msg.endswith('seconds'))
    logger = logging.getLogger('asyncio')
    filter = Filter()
    logger.addFilter(filter)
    try:
        yield
    finally:
        logger.removeFilter(filter)