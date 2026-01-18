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
def block_to_schedule(block: ScheduleBlock) -> Schedule:
    """Convert ``ScheduleBlock`` to ``Schedule``.

    Args:
        block: A ``ScheduleBlock`` to convert.

    Returns:
        Scheduled pulse program.

    Raises:
        UnassignedDurationError: When any instruction duration is not assigned.
        PulseError: When the alignment context duration is shorter than the schedule duration.

    .. note:: This transform may insert barriers in between contexts.
    """
    if not block.is_schedulable():
        raise UnassignedDurationError('All instruction durations should be assigned before creating `Schedule`.Please check `.parameters` to find unassigned parameter objects.')
    schedule = Schedule.initialize_from(block)
    for op_data in block.blocks:
        if isinstance(op_data, ScheduleBlock):
            context_schedule = block_to_schedule(op_data)
            if hasattr(op_data.alignment_context, 'duration'):
                post_buffer = op_data.alignment_context.duration - context_schedule.duration
                if post_buffer < 0:
                    raise PulseError(f'ScheduleBlock {op_data.name} has longer duration than the specified context duration {context_schedule.duration} > {op_data.duration}.')
            else:
                post_buffer = 0
            schedule.append(context_schedule, inplace=True)
            if post_buffer > 0:
                context_boundary = instructions.RelativeBarrier(*op_data.channels)
                schedule.append(context_boundary.shift(post_buffer), inplace=True)
        else:
            schedule.append(op_data, inplace=True)
    return block.alignment_context.align(schedule)