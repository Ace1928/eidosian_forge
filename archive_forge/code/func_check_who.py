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
def check_who(meth):

    @functools.wraps(meth)
    def wrapper(self, job, who, *args, **kwargs):
        if not isinstance(who, str):
            raise TypeError('Job applicant must be a string type')
        if len(who) == 0:
            raise ValueError('Job applicant must be non-empty')
        return meth(self, job, who, *args, **kwargs)
    return wrapper