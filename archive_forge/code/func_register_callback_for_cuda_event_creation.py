import logging
from typing import Callable, Generic, List
from typing_extensions import ParamSpec  # Python 3.10+
def register_callback_for_cuda_event_creation(cb: Callable[[int], None]) -> None:
    CUDAEventCreationCallbacks.add_callback(cb)