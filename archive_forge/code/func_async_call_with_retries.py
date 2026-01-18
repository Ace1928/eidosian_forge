import enum
import functools
import random
import threading
import time
import grpc
from tensorboard import version
from tensorboard.util import tb_logging
def async_call_with_retries(api_method, request, clock=None):
    """Initiate an asynchronous call to a gRPC stub, with retry logic.

    This is similar to the `async_call` API, except that the call is handled
    asynchronously, and the completion may be handled by another thread. The
    caller must provide a `done_callback` argument which will handle the
    result or exception rising from the gRPC completion.

    Retries are handled with jittered exponential backoff to spread out failures
    due to request spikes.

    This only supports unary-unary RPCs: i.e., no streaming on either end.

    Args:
      api_method: Callable for the API method to invoke.
      request: Request protocol buffer to pass to the API method.
      clock: an interface object supporting `time()` and `sleep()` methods
        like the standard `time` module; if not passed, uses the normal module.

    Returns:
      An `AsyncCallFuture` which will encapsulate the `grpc.Future`
      corresponding to the gRPC call which either completes successfully or
      represents the final try.
    """
    if clock is None:
        clock = time
    logger.debug('Async RPC call %s with request: %r', api_method, request)
    completion_event = threading.Event()
    async_future = AsyncCallFuture(completion_event)

    def async_call(handler):
        """Invokes the gRPC future and orchestrates it via the AsyncCallFuture."""
        future = api_method.future(request, timeout=_GRPC_DEFAULT_TIMEOUT_SECS, metadata=version_metadata())
        async_future._set_active_future(future)
        future.add_done_callback(handler)

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
    async_call(functools.partial(retry_handler, num_attempts=1))
    return async_future