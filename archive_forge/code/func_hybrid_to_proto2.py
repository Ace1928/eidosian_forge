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
def hybrid_to_proto2(message, body):
    """Create a fresh protocol 2 message from a hybrid protocol 1/2 message."""
    try:
        args, kwargs = (body.get('args', ()), body.get('kwargs', {}))
        kwargs.items
    except KeyError:
        raise InvalidTaskError('Message does not have args/kwargs')
    except AttributeError:
        raise InvalidTaskError('Task keyword arguments must be a mapping')
    headers = {'lang': body.get('lang'), 'task': body.get('task'), 'id': body.get('id'), 'root_id': body.get('root_id'), 'parent_id': body.get('parent_id'), 'group': body.get('group'), 'meth': body.get('meth'), 'shadow': body.get('shadow'), 'eta': body.get('eta'), 'expires': body.get('expires'), 'retries': body.get('retries', 0), 'timelimit': body.get('timelimit', (None, None)), 'argsrepr': body.get('argsrepr'), 'kwargsrepr': body.get('kwargsrepr'), 'origin': body.get('origin')}
    headers.update(message.headers or {})
    embed = {'callbacks': body.get('callbacks'), 'errbacks': body.get('errbacks'), 'chord': body.get('chord'), 'chain': None}
    return ((args, kwargs, embed), headers, True, body.get('utc', True))