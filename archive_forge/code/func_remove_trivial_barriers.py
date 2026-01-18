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
def remove_trivial_barriers(schedule: Schedule) -> Schedule:
    """Remove trivial barriers with 0 or 1 channels.

    Args:
        schedule: A schedule to remove trivial barriers.

    Returns:
        schedule: A schedule without trivial barriers
    """

    def filter_func(inst):
        return isinstance(inst[1], directives.RelativeBarrier) and len(inst[1].channels) < 2
    return schedule.exclude(filter_func)