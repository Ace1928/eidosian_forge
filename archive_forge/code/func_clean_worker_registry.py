from typing import TYPE_CHECKING, Optional, Set
from rq.utils import split_list
from .utils import as_text
def clean_worker_registry(queue: 'Queue'):
    """Delete invalid worker keys in registry.

    Args:
        queue (Queue): The Queue
    """
    keys = list(get_keys(queue))
    with queue.connection.pipeline() as pipeline:
        for key in keys:
            pipeline.exists(key)
        results = pipeline.execute()
        invalid_keys = []
        for i, key_exists in enumerate(results):
            if not key_exists:
                invalid_keys.append(keys[i])
        if invalid_keys:
            for invalid_subset in split_list(invalid_keys, MAX_KEYS):
                pipeline.srem(WORKERS_BY_QUEUE_KEY % queue.name, *invalid_subset)
                pipeline.srem(REDIS_WORKER_KEYS, *invalid_subset)
                pipeline.execute()