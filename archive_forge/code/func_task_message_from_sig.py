import numbers
from datetime import datetime, timedelta
from typing import Any, Mapping, Sequence  # noqa
from unittest.mock import Mock
from celery import Celery  # noqa
from celery.canvas import Signature  # noqa
def task_message_from_sig(app, sig, utc=True, TaskMessage=TaskMessage):
    """Create task message from :class:`celery.Signature`.

    Example:
        >>> m = task_message_from_sig(app, add.s(2, 2))
        >>> amqp_client.basic_publish(m, exchange='ex', routing_key='rkey')
    """
    sig.freeze()
    callbacks = sig.options.pop('link', None)
    errbacks = sig.options.pop('link_error', None)
    countdown = sig.options.pop('countdown', None)
    if countdown:
        eta = app.now() + timedelta(seconds=countdown)
    else:
        eta = sig.options.pop('eta', None)
    if eta and isinstance(eta, datetime):
        eta = eta.isoformat()
    expires = sig.options.pop('expires', None)
    if expires and isinstance(expires, numbers.Real):
        expires = app.now() + timedelta(seconds=expires)
    if expires and isinstance(expires, datetime):
        expires = expires.isoformat()
    return TaskMessage(sig.task, id=sig.id, args=sig.args, kwargs=sig.kwargs, callbacks=[dict(s) for s in callbacks] if callbacks else None, errbacks=[dict(s) for s in errbacks] if errbacks else None, eta=eta, expires=expires, utc=utc, **sig.options)