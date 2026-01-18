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
def append_schedule(schedule: Schedule | ScheduleBlock):
    """Call a schedule by appending to the active builder's context block.

    Args:
        schedule: Schedule or ScheduleBlock to append.
    """
    _active_builder().append_subroutine(schedule)