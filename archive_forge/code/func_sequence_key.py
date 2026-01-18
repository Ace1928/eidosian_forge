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
@misc.cachedproperty
def sequence_key(self):
    """Key where a integer will be stored (used to sequence jobs)."""
    return self.join(b'sequence')