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
def mark_as_failure(self, task_id, exc, traceback=None, request=None, store_result=True, call_errbacks=True, state=states.FAILURE):
    """Mark task as executed with failure."""
    if store_result:
        self.store_result(task_id, exc, state, traceback=traceback, request=request)
    if request:
        if request.chord:
            self.on_chord_part_return(request, state, exc)
        try:
            chain_data = iter(request.chain)
        except (AttributeError, TypeError):
            chain_data = tuple()
        for chain_elem in chain_data:
            chain_elem_ctx = Context(chain_elem)
            chain_elem_ctx.update(chain_elem_ctx.options)
            chain_elem_ctx.id = chain_elem_ctx.options.get('task_id')
            chain_elem_ctx.group = chain_elem_ctx.options.get('group_id')
            if store_result and state in states.PROPAGATE_STATES and (chain_elem_ctx.task_id is not None):
                self.store_result(chain_elem_ctx.task_id, exc, state, traceback=traceback, request=chain_elem_ctx)
            if 'chord' in chain_elem_ctx.options:
                self.on_chord_part_return(chain_elem_ctx, state, exc)
        if call_errbacks and request.errbacks:
            self._call_task_errbacks(request, exc, traceback)