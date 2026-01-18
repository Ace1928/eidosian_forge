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
@contextlib.contextmanager
def skip_valueerror():
    try:
        yield
    except ValueError:
        pytest.skip()