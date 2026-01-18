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
def test_journalhandler_info():
    record = logging.LogRecord('test-logger', logging.INFO, 'testpath', 1, 'test', None, None)
    sender = MockSender()
    kw = {'X': 3, 'X3': 4, 'sender_function': sender.send}
    handler = journal.JournalHandler(logging.INFO, **kw)
    handler.emit(record)
    assert len(sender.buf) == 1
    assert 'X=3' in sender.buf[0]
    assert 'X3=4' in sender.buf[0]
    sender = MockSender()
    handler = journal.JournalHandler.with_args({'level': logging.INFO, 'X': 3, 'X3': 4, 'sender_function': sender.send})
    handler.emit(record)
    assert len(sender.buf) == 1
    assert 'X=3' in sender.buf[0]
    assert 'X3=4' in sender.buf[0]
    journal.JournalHandler.with_args()