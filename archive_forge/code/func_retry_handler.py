import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def retry_handler(future, num_attempts):
    e = future.exception()
    if e is None:
        completion_event.set()
        return
    else:
        logger.info('RPC call %s got error %s', api_method, e)
        if e.code() not in _GRPC_RETRYABLE_STATUS_CODES:
            completion_event.set()
            return
        if num_attempts >= _GRPC_RETRY_MAX_ATTEMPTS:
            completion_event.set()
            return
        backoff_secs = _compute_backoff_seconds(num_attempts)
        clock.sleep(backoff_secs)
        async_call(functools.partial(retry_handler, num_attempts=num_attempts + 1))