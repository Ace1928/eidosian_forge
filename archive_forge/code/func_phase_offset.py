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
@contextmanager
def phase_offset(phase: float, *channels: chans.PulseChannel) -> Generator[None, None, None]:
    """Shift the phase of input channels on entry into context and undo on exit.

    Examples:

    .. code-block::

        import math

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build() as pulse_prog:
            with pulse.phase_offset(math.pi, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

    Args:
        phase: Amount of phase offset in radians.
        channels: Channels to offset phase of.

    Yields:
        None
    """
    for channel in channels:
        shift_phase(phase, channel)
    try:
        yield
    finally:
        for channel in channels:
            shift_phase(-phase, channel)