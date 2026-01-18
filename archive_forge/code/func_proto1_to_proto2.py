import logging
from kombu.asynchronous.timer import to_timestamp
from celery import signals
from celery.app import trace as _app_trace
from celery.exceptions import InvalidTaskError
from celery.utils.imports import symbol_by_name
from celery.utils.log import get_logger
from celery.utils.saferepr import saferepr
from celery.utils.time import timezone
from .request import create_request_cls
from .state import task_reserved
def proto1_to_proto2(message, body):
    """Convert Task message protocol 1 arguments to protocol 2.

    Returns:
        Tuple: of ``(body, headers, already_decoded_status, utc)``
    """
    try:
        args, kwargs = (body.get('args', ()), body.get('kwargs', {}))
        kwargs.items
    except KeyError:
        raise InvalidTaskError('Message does not have args/kwargs')
    except AttributeError:
        raise InvalidTaskError('Task keyword arguments must be a mapping')
    body.update(argsrepr=saferepr(args), kwargsrepr=saferepr(kwargs), headers=message.headers)
    try:
        body['group'] = body['taskset']
    except KeyError:
        pass
    embed = {'callbacks': body.get('callbacks'), 'errbacks': body.get('errbacks'), 'chord': body.get('chord'), 'chain': None}
    return ((args, kwargs, embed), body, True, body.get('utc', True))