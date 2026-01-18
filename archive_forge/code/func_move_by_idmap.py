import socket
from functools import partial
from itertools import cycle, islice
from kombu import Queue, eventloop
from kombu.common import maybe_declare
from kombu.utils.encoding import ensure_bytes
from celery.app import app_or_default
from celery.utils.nodenames import worker_direct
from celery.utils.text import str_to_list
def move_by_idmap(map, **kwargs):
    """Move tasks by matching from a ``task_id: queue`` mapping.

    Where ``queue`` is a queue to move the task to.

    Example:
        >>> move_by_idmap({
        ...     '5bee6e82-f4ac-468e-bd3d-13e8600250bc': Queue('name'),
        ...     'ada8652d-aef3-466b-abd2-becdaf1b82b3': Queue('name'),
        ...     '3a2b140d-7db1-41ba-ac90-c36a0ef4ab1f': Queue('name')},
        ...   queues=['hipri'])
    """

    def task_id_in_map(body, message):
        return map.get(message.properties['correlation_id'])
    return move(task_id_in_map, limit=len(map), **kwargs)