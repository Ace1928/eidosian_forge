import inspect
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type, Union
import torch
from torch._streambase import _EventBase, _StreamBase
@staticmethod
def get_raw_stream():
    raise NotImplementedError()