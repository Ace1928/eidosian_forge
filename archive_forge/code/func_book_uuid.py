import abc
import collections
import contextlib
import functools
import time
import enum
from oslo_utils import timeutils
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions as excp
from taskflow import states
from taskflow.types import notifier
from taskflow.utils import iter_utils
@property
def book_uuid(self):
    """UUID of logbook associated with this job.

        If no logbook is associated with this job, this property is None.
        """
    if self._book is not None:
        return self._book.uuid
    else:
        return self._book_data.get('uuid')