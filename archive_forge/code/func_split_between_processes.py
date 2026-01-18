from __future__ import annotations
import logging
import math
import os
import threading
import warnings
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional
import torch
from .utils import (
from .utils.dataclasses import SageMakerDistributedType
@contextmanager
def split_between_processes(self, inputs: list | tuple | dict | torch.Tensor, apply_padding: bool=False):
    """
        Splits `input` between `self.num_processes` quickly and can be then used on that process. Useful when doing
        distributed inference, such as with different prompts.

        Note that when using a `dict`, all keys need to have the same number of elements.

        Args:
            inputs (`list`, `tuple`, `torch.Tensor`, or `dict` of `list`/`tuple`/`torch.Tensor`):
                The input to split between processes.
            apply_padding (`bool`, `optional`, defaults to `False`):
                Whether to apply padding by repeating the last element of the input so that all processes have the same
                number of elements. Useful when trying to perform actions such as `gather()` on the outputs or passing
                in less inputs than there are processes. If so, just remember to drop the padded elements afterwards.


        Example:

        ```python
        # Assume there are two processes
        from accelerate.state import AcceleratorState

        state = AcceleratorState()
        with state.split_between_processes(["A", "B", "C"]) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C"]

        with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
            print(inputs)
        # Process 0
        ["A", "B"]
        # Process 1
        ["C", "C"]
        ```
        """
    with PartialState().split_between_processes(inputs, apply_padding=apply_padding) as inputs:
        yield inputs