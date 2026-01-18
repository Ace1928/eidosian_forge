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
@instruction_filter.register
def handle_instruction(inst: Instruction) -> bool:
    """Filter instruction.

        Args:
            inst: Instruction

        Returns:
            If instruction matches with condition.
        """
    return isinstance(inst, tuple(types))