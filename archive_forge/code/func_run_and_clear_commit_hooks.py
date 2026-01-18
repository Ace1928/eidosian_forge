import _thread
import copy
import datetime
import logging
import threading
import time
import warnings
import zoneinfo
from collections import deque
from contextlib import contextmanager
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured
from django.db import DEFAULT_DB_ALIAS, DatabaseError, NotSupportedError
from django.db.backends import utils
from django.db.backends.base.validation import BaseDatabaseValidation
from django.db.backends.signals import connection_created
from django.db.backends.utils import debug_transaction
from django.db.transaction import TransactionManagementError
from django.db.utils import DatabaseErrorWrapper
from django.utils.asyncio import async_unsafe
from django.utils.functional import cached_property
def run_and_clear_commit_hooks(self):
    self.validate_no_atomic_block()
    current_run_on_commit = self.run_on_commit
    self.run_on_commit = []
    while current_run_on_commit:
        _, func, robust = current_run_on_commit.pop(0)
        if robust:
            try:
                func()
            except Exception as e:
                logger.error(f'Error calling {func.__qualname__} in on_commit() during transaction (%s).', e, exc_info=True)
        else:
            func()