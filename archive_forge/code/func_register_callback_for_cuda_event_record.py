import logging
from typing import Callable, Generic, List
from typing_extensions import ParamSpec  # Python 3.10+
def register_callback_for_cuda_event_record(cb: Callable[[int, int], None]) -> None:
    CUDAEventRecordCallbacks.add_callback(cb)