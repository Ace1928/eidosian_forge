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
def test_journalhandler_init_exception():
    kw = {' X  ': 3}
    with pytest.raises(ValueError):
        journal.JournalHandler(**kw)
    with pytest.raises(ValueError):
        journal.JournalHandler.with_args(kw)