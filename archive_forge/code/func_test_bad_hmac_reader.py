import asyncore
import errno
import socket
import threading
from taskflow.engines.action_engine import process_executor as pu
from taskflow import task
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils as test_utils
def test_bad_hmac_reader(self):
    r = pu.Reader(b'secret-2', lambda ident, capture_func: capture_func())
    in_data = b''.join(pu._encode_message(b'secret', ['hi'], b'me'))
    self.assertRaises(pu.BadHmacValueError, r.feed, in_data)