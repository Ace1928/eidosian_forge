from __future__ import annotations
import contextvars
import functools
import itertools
import sys
import uuid
import warnings
from collections.abc import Generator, Callable, Iterable
from contextlib import contextmanager
from functools import singledispatchmethod
from typing import TypeVar, ContextManager, TypedDict, Union, Optional, Dict
import numpy as np
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import (
from qiskit.providers.backend import BackendV2
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock
from qiskit.pulse.transforms.alignments import AlignmentKind
def append_subroutine(self, subroutine: Schedule | ScheduleBlock):
    """Append a :class:`ScheduleBlock` to the builder's context schedule.

        This operation doesn't create a reference. Subroutine is directly
        appended to current context schedule.

        Args:
            subroutine: ScheduleBlock to append to the current context block.

        Raises:
            PulseError: When subroutine is not Schedule nor ScheduleBlock.
        """
    if not isinstance(subroutine, (ScheduleBlock, Schedule)):
        raise exceptions.PulseError(f"'{subroutine.__class__.__name__}' is not valid data format in the builder. 'Schedule' and 'ScheduleBlock' can be appended to the builder context.")
    if len(subroutine) == 0:
        return
    if isinstance(subroutine, Schedule):
        subroutine = self._naive_typecast_schedule(subroutine)
    self._context_stack[-1].append(subroutine)