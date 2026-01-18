import contextlib
import datetime
import functools
import re
import string
import threading
import time
import fasteners
import msgpack
from oslo_serialization import msgpackutils
from oslo_utils import excutils
from oslo_utils import strutils
from oslo_utils import timeutils
from oslo_utils import uuidutils
from redis import exceptions as redis_exceptions
from redis import sentinel
from taskflow import exceptions as exc
from taskflow.jobs import base
from taskflow import logging
from taskflow import states
from taskflow.utils import misc
from taskflow.utils import redis_utils as ru
def iterjobs(self, only_unclaimed=False, ensure_fresh=False):
    return base.JobBoardIterator(self, LOG, only_unclaimed=only_unclaimed, ensure_fresh=ensure_fresh, board_fetch_func=lambda ensure_fresh: self._fetch_jobs())