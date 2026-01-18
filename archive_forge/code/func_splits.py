import calendar
import datetime
import functools
import logging
import time
import iso8601
from oslo_utils import reflection
@property
def splits(self):
    """Accessor to all/any splits that have been captured."""
    return self._splits