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
def disable_logger():
    """Context manager to disable asyncio logger.

    For example, it can be used to ignore warnings in debug mode.
    """
    old_level = asyncio.log.logger.level
    try:
        asyncio.log.logger.setLevel(logging.CRITICAL + 1)
        yield
    finally:
        asyncio.log.logger.setLevel(old_level)