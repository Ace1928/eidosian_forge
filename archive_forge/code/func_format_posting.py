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
def format_posting(uuid, name, created_on=None, last_modified=None, details=None, book=None, priority=JobPriority.NORMAL):
    posting = {'uuid': uuid, 'name': name, 'priority': priority.value}
    if created_on is not None:
        posting['created_on'] = created_on
    if last_modified is not None:
        posting['last_modified'] = last_modified
    if details:
        posting['details'] = details
    else:
        posting['details'] = {}
    if book is not None:
        posting['book'] = {'name': book.name, 'uuid': book.uuid}
    return posting