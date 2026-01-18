from __future__ import absolute_import
import email
import logging
import re
import time
import warnings
from collections import namedtuple
from itertools import takewhile
from ..exceptions import (
from ..packages import six
def is_exhausted(self):
    """Are we out of retries?"""
    retry_counts = (self.total, self.connect, self.read, self.redirect, self.status, self.other)
    retry_counts = list(filter(None, retry_counts))
    if not retry_counts:
        return False
    return min(retry_counts) < 0