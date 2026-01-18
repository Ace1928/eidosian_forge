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
def send_task_message(producer, name, message, exchange=None, routing_key=None, queue=None, event_dispatcher=None, retry=None, retry_policy=None, serializer=None, delivery_mode=None, compression=None, declare=None, headers=None, exchange_type=None, **kwargs):
    retry = default_retry if retry is None else retry
    headers2, properties, body, sent_event = message
    if headers:
        headers2.update(headers)
    if kwargs:
        properties.update(kwargs)
    qname = queue
    if queue is None and exchange is None:
        queue = default_queue
    if queue is not None:
        if isinstance(queue, str):
            qname, queue = (queue, queues[queue])
        else:
            qname = queue.name
    if delivery_mode is None:
        try:
            delivery_mode = queue.exchange.delivery_mode
        except AttributeError:
            pass
        delivery_mode = delivery_mode or default_delivery_mode
    if exchange_type is None:
        try:
            exchange_type = queue.exchange.type
        except AttributeError:
            exchange_type = 'direct'
    if (not exchange or not routing_key) and exchange_type == 'direct':
        exchange, routing_key = ('', qname)
    elif exchange is None:
        exchange = queue.exchange.name or default_exchange
        routing_key = routing_key or queue.routing_key or default_rkey
    if declare is None and queue and (not isinstance(queue, Broadcast)):
        declare = [queue]
    retry = default_retry if retry is None else retry
    _rp = dict(default_policy, **retry_policy) if retry_policy else default_policy
    if before_receivers:
        send_before_publish(sender=name, body=body, exchange=exchange, routing_key=routing_key, declare=declare, headers=headers2, properties=properties, retry_policy=retry_policy)
    ret = producer.publish(body, exchange=exchange, routing_key=routing_key, serializer=serializer or default_serializer, compression=compression or default_compressor, retry=retry, retry_policy=_rp, delivery_mode=delivery_mode, declare=declare, headers=headers2, **properties)
    if after_receivers:
        send_after_publish(sender=name, body=body, headers=headers2, exchange=exchange, routing_key=routing_key)
    if sent_receivers:
        if isinstance(body, tuple):
            send_task_sent(sender=name, task_id=headers2['id'], task=name, args=body[0], kwargs=body[1], eta=headers2['eta'], taskset=headers2['group'])
        else:
            send_task_sent(sender=name, task_id=body['id'], task=name, args=body['args'], kwargs=body['kwargs'], eta=body['eta'], taskset=body['taskset'])
    if sent_event:
        evd = event_dispatcher or default_evd
        exname = exchange
        if isinstance(exname, Exchange):
            exname = exname.name
        sent_event.update({'queue': qname, 'exchange': exname, 'routing_key': routing_key})
        evd.publish('task-sent', sent_event, producer, retry=retry, retry_policy=retry_policy)
    return ret