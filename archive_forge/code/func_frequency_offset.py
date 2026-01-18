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
def frequency_offset(frequency: float, *channels: chans.PulseChannel, compensate_phase: bool=False) -> Generator[None, None, None]:
    """Shift the frequency of inputs channels on entry into context and undo on exit.

    Examples:

    .. code-block:: python
        :emphasize-lines: 7, 16

        from qiskit import pulse

        d0 = pulse.DriveChannel(0)

        with pulse.build(backend) as pulse_prog:
            # shift frequency by 1GHz
            with pulse.frequency_offset(1e9, d0):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 3

        with pulse.build(backend) as pulse_prog:
            # Shift frequency by 1GHz.
            # Undo accumulated phase in the shifted frequency frame
            # when exiting the context.
            with pulse.frequency_offset(1e9, d0, compensate_phase=True):
                pulse.play(pulse.Constant(10, 1.0), d0)

        assert len(pulse_prog.instructions) == 4

    Args:
        frequency: Amount of frequency offset in Hz.
        channels: Channels to offset frequency of.
        compensate_phase: Compensate for accumulated phase accumulated with
            respect to the channels' frame at its initial frequency.

    Yields:
        None
    """
    builder = _active_builder()
    t0 = builder.get_context().duration
    for channel in channels:
        shift_frequency(frequency, channel)
    try:
        yield
    finally:
        if compensate_phase:
            duration = builder.get_context().duration - t0
            accumulated_phase = 2 * np.pi * (duration * builder.get_dt() * frequency % 1)
            for channel in channels:
                shift_phase(-accumulated_phase, channel)
        for channel in channels:
            shift_frequency(-frequency, channel)