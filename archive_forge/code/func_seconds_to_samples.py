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
def seconds_to_samples(seconds: float | np.ndarray) -> int | np.ndarray:
    """Obtain the number of samples that will elapse in ``seconds`` on the
    active backend.

    Rounds down.

    Args:
        seconds: Time in seconds to convert to samples.

    Returns:
        The number of samples for the time to elapse
    """
    dt = _active_builder().get_dt()
    if isinstance(seconds, np.ndarray):
        return (seconds / dt).astype(int)
    return int(seconds / dt)