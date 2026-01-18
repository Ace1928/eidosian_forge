from __future__ import annotations
import abc
from functools import singledispatch
from collections.abc import Iterable
from typing import Callable, Any, List
import numpy as np
from qiskit.pulse import Schedule, ScheduleBlock, Instruction
from qiskit.pulse.channels import Channel
from qiskit.pulse.schedule import Interval
from qiskit.pulse.exceptions import PulseError
@filter_instructions.register
def handle_schedule(sched: Schedule, filters: List[Callable[..., bool]], negate: bool=False, recurse_subroutines: bool=True) -> Schedule:
    """A filtering function that takes a schedule and returns a schedule consisting of
    filtered instructions.

    Args:
        sched: A pulse schedule to be filtered.
        filters: List of callback functions that take an instruction and return boolean.
        negate: Set `True` to accept an instruction if a filter function returns `False`.
            Otherwise the instruction is accepted when the filter function returns `False`.
        recurse_subroutines: Set `True` to individually filter instructions inside of a subroutine
            defined by the :py:class:`~qiskit.pulse.instructions.Call` instruction.

    Returns:
        Filtered pulse schedule.
    """
    from qiskit.pulse.transforms import flatten, inline_subroutines
    target_sched = flatten(sched)
    if recurse_subroutines:
        target_sched = inline_subroutines(target_sched)
    time_inst_tuples = np.array(target_sched.instructions)
    valid_insts = np.ones(len(time_inst_tuples), dtype=bool)
    for filt in filters:
        valid_insts = np.logical_and(valid_insts, np.array(list(map(filt, time_inst_tuples))))
    if negate and len(filters) > 0:
        valid_insts = ~valid_insts
    filter_schedule = Schedule.initialize_from(sched)
    for time, inst in time_inst_tuples[valid_insts]:
        filter_schedule.insert(time, inst, inplace=True)
    return filter_schedule