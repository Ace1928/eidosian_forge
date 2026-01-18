from __future__ import annotations
import csv
import hashlib
import os.path
import re
import stat
import time
from io import StringIO, TextIOWrapper
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo
from wheel.cli import WheelError
from wheel.util import log, urlsafe_b64decode, urlsafe_b64encode
def get_zipinfo_datetime(timestamp=None):
    timestamp = int(os.environ.get('SOURCE_DATE_EPOCH', timestamp or time.time()))
    timestamp = max(timestamp, MINIMUM_TIMESTAMP)
    return time.gmtime(timestamp)[0:6]