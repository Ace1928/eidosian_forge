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
def compress_pulses(schedules: list[Schedule]) -> list[Schedule]:
    """Optimization pass to replace identical pulses.

    Args:
        schedules: Schedules to compress.

    Returns:
        Compressed schedules.
    """
    existing_pulses: list[Pulse] = []
    new_schedules = []
    for schedule in schedules:
        new_schedule = Schedule.initialize_from(schedule)
        for time, inst in schedule.instructions:
            if isinstance(inst, instructions.Play):
                if inst.pulse in existing_pulses:
                    idx = existing_pulses.index(inst.pulse)
                    identical_pulse = existing_pulses[idx]
                    new_schedule.insert(time, instructions.Play(identical_pulse, inst.channel, inst.name), inplace=True)
                else:
                    existing_pulses.append(inst.pulse)
                    new_schedule.insert(time, inst, inplace=True)
            else:
                new_schedule.insert(time, inst, inplace=True)
        new_schedules.append(new_schedule)
    return new_schedules