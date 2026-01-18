from typing import Union, Iterable, Tuple
from qiskit.pulse.instructions import Instruction
from qiskit.pulse.schedule import ScheduleBlock, Schedule
from qiskit.pulse.transforms import canonicalization
def target_qobj_transform(sched: Union[ScheduleBlock, Schedule, InstructionSched, Iterable[InstructionSched]], remove_directives: bool=True) -> Schedule:
    """A basic pulse program transformation for OpenPulse API execution.

    Args:
        sched: Input program to transform.
        remove_directives: Set `True` to remove compiler directives.

    Returns:
        Transformed program for execution.
    """
    if not isinstance(sched, Schedule):
        if isinstance(sched, ScheduleBlock):
            sched = canonicalization.block_to_schedule(sched)
        else:
            sched = Schedule(*_format_schedule_component(sched))
    sched = canonicalization.inline_subroutines(sched)
    sched = canonicalization.flatten(sched)
    if remove_directives:
        sched = canonicalization.remove_directives(sched)
    return sched