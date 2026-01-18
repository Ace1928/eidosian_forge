import collections
import contextlib
import io
import logging
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from http.server import HTTPServer
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer
from . import base_events
from . import events
from . import futures
from . import selectors
from . import tasks
from .coroutines import coroutine
from .log import logger
def assert_writer(self, fd, callback, *args):
    assert fd in self.writers, 'fd {} is not registered'.format(fd)
    handle = self.writers[fd]
    assert handle._callback == callback, '{!r} != {!r}'.format(handle._callback, callback)
    assert handle._args == args, '{!r} != {!r}'.format(handle._args, args)