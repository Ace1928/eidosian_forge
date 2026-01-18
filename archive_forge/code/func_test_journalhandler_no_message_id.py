from __future__ import print_function
import contextlib
import datetime
import errno
import logging
import os
import time
import uuid
import sys
import traceback
from systemd import journal, id128
from systemd.journal import _make_line
import pytest
def test_journalhandler_no_message_id():
    record = logging.LogRecord('test-logger', logging.INFO, 'testpath', 1, 'test', None, None)
    sender = MockSender()
    handler = journal.JournalHandler(logging.INFO, sender_function=sender.send)
    handler.emit(record)
    assert len(sender.buf) == 1
    assert all((not m.startswith('MESSAGE_ID=') for m in sender.buf[0]))