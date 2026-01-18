import fnmatch
import re
from collections import OrderedDict
from collections.abc import Mapping
from kombu import Queue
from celery.exceptions import QueueNotFound
from celery.utils.collections import lpmerge
from celery.utils.functional import maybe_evaluate, mlazy
from celery.utils.imports import symbol_by_name
def expand_destination(self, route):
    if isinstance(route, str):
        queue, route = (route, {})
    else:
        queue = route.pop('queue', None)
    if queue:
        if isinstance(queue, Queue):
            route['queue'] = queue
        else:
            try:
                route['queue'] = self.queues[queue]
            except KeyError:
                raise QueueNotFound(f'Queue {queue!r} missing from task_queues')
    return route