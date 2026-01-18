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
def remove_directives(schedule: Schedule) -> Schedule:
    """Remove directives.

    Args:
        schedule: A schedule to remove compiler directives.

    Returns:
        A schedule without directives.
    """
    return schedule.exclude(instruction_types=[directives.Directive])