import numbers
from collections import namedtuple
from collections.abc import Mapping
from datetime import timedelta
from weakref import WeakValueDictionary
from kombu import Connection, Consumer, Exchange, Producer, Queue, pools
from kombu.common import Broadcast
from kombu.utils.functional import maybe_list
from kombu.utils.objects import cached_property
from celery import signals
from celery.utils.nodenames import anon_nodename
from celery.utils.saferepr import saferepr
from celery.utils.text import indent as textindent
from celery.utils.time import maybe_make_aware
from . import routes as _routes
def as_task_v1(self, task_id, name, args=None, kwargs=None, countdown=None, eta=None, group_id=None, group_index=None, expires=None, retries=0, chord=None, callbacks=None, errbacks=None, reply_to=None, time_limit=None, soft_time_limit=None, create_sent_event=False, root_id=None, parent_id=None, shadow=None, now=None, timezone=None, **compat_kwargs):
    args = args or ()
    kwargs = kwargs or {}
    utc = self.utc
    if not isinstance(args, (list, tuple)):
        raise TypeError('task args must be a list or tuple')
    if not isinstance(kwargs, Mapping):
        raise TypeError('task keyword arguments must be a mapping')
    if countdown:
        self._verify_seconds(countdown, 'countdown')
        now = now or self.app.now()
        eta = now + timedelta(seconds=countdown)
    if isinstance(expires, numbers.Real):
        self._verify_seconds(expires, 'expires')
        now = now or self.app.now()
        expires = now + timedelta(seconds=expires)
    eta = eta and eta.isoformat()
    expires = expires and expires.isoformat()
    return task_message(headers={}, properties={'correlation_id': task_id, 'reply_to': reply_to or ''}, body={'task': name, 'id': task_id, 'args': args, 'kwargs': kwargs, 'group': group_id, 'group_index': group_index, 'retries': retries, 'eta': eta, 'expires': expires, 'utc': utc, 'callbacks': callbacks, 'errbacks': errbacks, 'timelimit': (time_limit, soft_time_limit), 'taskset': group_id, 'chord': chord}, sent_event={'uuid': task_id, 'name': name, 'args': saferepr(args), 'kwargs': saferepr(kwargs), 'retries': retries, 'eta': eta, 'expires': expires} if create_sent_event else None)