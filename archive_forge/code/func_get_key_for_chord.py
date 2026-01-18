import sys
import time
import warnings
from collections import namedtuple
from datetime import datetime, timedelta, timezone
from functools import partial
from weakref import WeakValueDictionary
from billiard.einfo import ExceptionInfo
from kombu.serialization import dumps, loads, prepare_accept_content
from kombu.serialization import registry as serializer_registry
from kombu.utils.encoding import bytes_to_str, ensure_bytes
from kombu.utils.url import maybe_sanitize_url
import celery.exceptions
from celery import current_app, group, maybe_signature, states
from celery._state import get_current_task
from celery.app.task import Context
from celery.exceptions import (BackendGetMetaError, BackendStoreError, ChordError, ImproperlyConfigured,
from celery.result import GroupResult, ResultBase, ResultSet, allow_join_result, result_from_tuple
from celery.utils.collections import BufferMap
from celery.utils.functional import LRUCache, arity_greater
from celery.utils.log import get_logger
from celery.utils.serialization import (create_exception_cls, ensure_serializable, get_pickleable_exception,
from celery.utils.time import get_exponential_backoff_interval
def get_key_for_chord(self, group_id, key=''):
    """Get the cache key for the chord waiting on group with given id."""
    if not group_id:
        raise ValueError(f'group_id must not be empty. Got {group_id} instead.')
    return self._get_key_for(self.chord_keyprefix, group_id, key)