import inspect
import os
import sys
import threading
import warnings
from collections import UserDict, defaultdict, deque
from datetime import datetime
from datetime import timezone as datetime_timezone
from operator import attrgetter
from click.exceptions import Exit
from dateutil.parser import isoparse
from kombu import pools
from kombu.clocks import LamportClock
from kombu.common import oid_from
from kombu.utils.compat import register_after_fork
from kombu.utils.objects import cached_property
from kombu.utils.uuid import uuid
from vine import starpromise
from celery import platforms, signals
from celery._state import (_announce_app_finalized, _deregister_app, _register_app, _set_current_app, _task_stack,
from celery.exceptions import AlwaysEagerIgnored, ImproperlyConfigured
from celery.loaders import get_loader_cls
from celery.local import PromiseProxy, maybe_evaluate
from celery.utils import abstract
from celery.utils.collections import AttributeDictMixin
from celery.utils.dispatch import Signal
from celery.utils.functional import first, head_from_fun, maybe_list
from celery.utils.imports import gen_task_name, instantiate, symbol_by_name
from celery.utils.log import get_logger
from celery.utils.objects import FallbackContext, mro_lookup
from celery.utils.time import maybe_make_aware, timezone, to_utc
from . import backends, builtins  # noqa
from .annotations import prepare as prepare_annotations
from .autoretry import add_autoretry_behaviour
from .defaults import DEFAULT_SECURITY_DIGEST, find_deprecated_settings
from .registry import TaskRegistry
from .utils import (AppPickler, Settings, _new_key_to_old, _old_key_to_new, _unpickle_app, _unpickle_app_v2, appstr,
def send_task(self, name, args=None, kwargs=None, countdown=None, eta=None, task_id=None, producer=None, connection=None, router=None, result_cls=None, expires=None, publisher=None, link=None, link_error=None, add_to_parent=True, group_id=None, group_index=None, retries=0, chord=None, reply_to=None, time_limit=None, soft_time_limit=None, root_id=None, parent_id=None, route_name=None, shadow=None, chain=None, task_type=None, replaced_task_nesting=0, **options):
    """Send task by name.

        Supports the same arguments as :meth:`@-Task.apply_async`.

        Arguments:
            name (str): Name of task to call (e.g., `"tasks.add"`).
            result_cls (AsyncResult): Specify custom result class.
        """
    parent = have_parent = None
    amqp = self.amqp
    task_id = task_id or uuid()
    producer = producer or publisher
    router = router or amqp.router
    conf = self.conf
    if conf.task_always_eager:
        warnings.warn(AlwaysEagerIgnored('task_always_eager has no effect on send_task'), stacklevel=2)
    ignore_result = options.pop('ignore_result', False)
    options = router.route(options, route_name or name, args, kwargs, task_type)
    if expires is not None:
        if isinstance(expires, datetime):
            expires_s = (maybe_make_aware(expires) - self.now()).total_seconds()
        elif isinstance(expires, str):
            expires_s = (maybe_make_aware(isoparse(expires)) - self.now()).total_seconds()
        else:
            expires_s = expires
        if expires_s < 0:
            logger.warning(f"""{task_id} has an expiration date in the past ({-expires_s}s ago).\nWe assume this is intended and so we have set the expiration date to 0 instead.\nAccording to RabbitMQ's documentation:\n"Setting the TTL to 0 causes messages to be expired upon reaching a queue unless they can be delivered to a consumer immediately."\nIf this was unintended, please check the code which published this task.""")
            expires_s = 0
        options['expiration'] = expires_s
    if not root_id or not parent_id:
        parent = self.current_worker_task
        if parent:
            if not root_id:
                root_id = parent.request.root_id or parent.request.id
            if not parent_id:
                parent_id = parent.request.id
            if conf.task_inherit_parent_priority:
                options.setdefault('priority', parent.request.delivery_info.get('priority'))
    message = amqp.create_task_message(task_id, name, args, kwargs, countdown, eta, group_id, group_index, expires, retries, chord, maybe_list(link), maybe_list(link_error), reply_to or self.thread_oid, time_limit, soft_time_limit, self.conf.task_send_sent_event, root_id, parent_id, shadow, chain, ignore_result=ignore_result, replaced_task_nesting=replaced_task_nesting, **options)
    stamped_headers = options.pop('stamped_headers', [])
    for stamp in stamped_headers:
        options.pop(stamp)
    if connection:
        producer = amqp.Producer(connection, auto_declare=False)
    with self.producer_or_acquire(producer) as P:
        with P.connection._reraise_as_library_errors():
            if not ignore_result:
                self.backend.on_task_call(P, task_id)
            amqp.send_task_message(P, name, message, **options)
    result = (result_cls or self.AsyncResult)(task_id)
    result.ignored = ignore_result
    if add_to_parent:
        if not have_parent:
            parent, have_parent = (self.current_worker_task, True)
        if parent:
            parent.add_trail(result)
    return result