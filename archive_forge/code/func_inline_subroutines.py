from __future__ import annotations
import typing
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Type
import numpy as np
from qiskit.pulse import channels as chans, exceptions, instructions
from qiskit.pulse.channels import ClassicalIOChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.exceptions import UnassignedDurationError
from qiskit.pulse.instruction_schedule_map import InstructionScheduleMap
from qiskit.pulse.instructions import directives
from qiskit.pulse.schedule import Schedule, ScheduleBlock, ScheduleComponent
def inline_subroutines(program: Schedule | ScheduleBlock) -> Schedule | ScheduleBlock:
    """Recursively remove call instructions and inline the respective subroutine instructions.

    Assigned parameter values, which are stored in the parameter table, are also applied.
    The subroutine is copied before the parameter assignment to avoid mutation problem.

    Args:
        program: A program which may contain the subroutine, i.e. ``Call`` instruction.

    Returns:
        A schedule without subroutine.

    Raises:
        PulseError: When input program is not valid data format.
    """
    if isinstance(program, Schedule):
        return _inline_schedule(program)
    elif isinstance(program, ScheduleBlock):
        return _inline_block(program)
    else:
        raise PulseError(f'Invalid program {program.__class__.__name__} is specified.')