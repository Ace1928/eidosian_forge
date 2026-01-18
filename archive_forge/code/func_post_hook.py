import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import Any, List, NamedTuple, Optional, Type
import torch
import torch.distributed as dist
def post_hook(self, is_last_joiner: bool) -> None:
    """
        Call hook after all processes have joined.

        It is passed an additional ``bool`` argument ``is_last_joiner``, which indicates if the rank is one of the last to join.

        Arguments:
            is_last_joiner (bool): ``True`` if the rank is one of the last to
                join; ``False`` otherwise.
        """
    ...