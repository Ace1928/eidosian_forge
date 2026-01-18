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
def deselect(self, exclude):
    """Deselect queues so that they won't be consumed from.

        Arguments:
            exclude (Sequence[str], str): Names of queues to avoid
                consuming from.
        """
    if exclude:
        exclude = maybe_list(exclude)
        if self._consume_from is None:
            return self.select((k for k in self if k not in exclude))
        for queue in exclude:
            self._consume_from.pop(queue, None)