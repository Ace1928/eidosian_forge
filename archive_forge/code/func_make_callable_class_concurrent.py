from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, List
import ray
from ray.data.block import BlockAccessor, CallableClass
def make_callable_class_concurrent(callable_cls: CallableClass) -> CallableClass:
    """Returns a thread-safe CallableClass with the same logic as the provided
    `callable_cls`.

    This function allows the usage of concurrent actors by safeguarding user logic
    behind a separate thread.

    This allows batch slicing and formatting to occur concurrently, to overlap with the
    user provided UDF.
    """

    class _Wrapper(callable_cls):

        def __init__(self, *args, **kwargs):
            self.thread_pool_executor = ThreadPoolExecutor(max_workers=1)
            super().__init__(*args, **kwargs)

        def __repr__(self):
            return super().__repr__()

        def __call__(self, *args, **kwargs):
            future = self.thread_pool_executor.submit(super().__call__, *args, **kwargs)
            return future.result()
    return _Wrapper