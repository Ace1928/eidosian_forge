from __future__ import annotations
import abc
import copy
import functools
import itertools
import multiprocessing as mp
import sys
import warnings
from collections.abc import Callable, Iterable
from typing import List, Tuple, Union, Dict, Any
import numpy as np
import rustworkx as rx
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.channels import Channel
from qiskit.pulse.exceptions import PulseError, UnassignedReferenceError
from qiskit.pulse.instructions import Instruction, Reference
from qiskit.pulse.utils import instruction_duration_validation
from qiskit.pulse.reference_manager import ReferenceManager
from qiskit.utils.multiprocessing import is_main_process
def assign_references(self, subroutine_dict: dict[str | tuple[str, ...], 'ScheduleBlock'], inplace: bool=True) -> 'ScheduleBlock':
    """Assign schedules to references.

        It is only capable of assigning a schedule block to immediate references
        which are directly referred within the current scope.
        Let's see following example:

        .. code-block:: python

            from qiskit import pulse

            with pulse.build() as subroutine:
                pulse.delay(10, pulse.DriveChannel(0))

            with pulse.build() as sub_prog:
                pulse.reference("A")

            with pulse.build() as main_prog:
                pulse.reference("B")

        In above example, the ``main_prog`` can refer to the subroutine "root::B" and the
        reference of "B" to program "A", i.e., "B::A", is not defined in the root namespace.
        This prevents breaking the reference "root::B::A" by the assignment of "root::B".
        For example, if a user could indirectly assign "root::B::A" from the root program,
        one can later assign another program to "root::B" that doesn't contain "A" within it.
        In this situation, a reference "root::B::A" would still live in
        the reference manager of the root.
        However, the subroutine "root::B::A" would no longer be used in the actual pulse program.
        To assign subroutine "A" to ``nested_prog`` as a nested subprogram of ``main_prog``,
        you must first assign "A" of the ``sub_prog``,
        and then assign the ``sub_prog`` to the ``main_prog``.

        .. code-block:: python

            sub_prog.assign_references({("A", ): nested_prog}, inplace=True)
            main_prog.assign_references({("B", ): sub_prog}, inplace=True)

        Alternatively, you can also write

        .. code-block:: python

            main_prog.assign_references({("B", ): sub_prog}, inplace=True)
            main_prog.references[("B", )].assign_references({"A": nested_prog}, inplace=True)

        Here :attr:`.references` returns a dict-like object, and you can
        mutably update the nested reference of the particular subroutine.

        .. note::

            Assigned programs are deep-copied to prevent an unexpected update.

        Args:
            subroutine_dict: A mapping from reference key to schedule block of the subroutine.
            inplace: Set ``True`` to override this instance with new subroutine.

        Returns:
            Schedule block with assigned subroutine.

        Raises:
            PulseError: When reference key is not defined in the current scope.
        """
    if not inplace:
        new_schedule = copy.deepcopy(self)
        return new_schedule.assign_references(subroutine_dict, inplace=True)
    for key, subroutine in subroutine_dict.items():
        if key not in self.references:
            unassigned_keys = ', '.join(map(repr, self.references.unassigned()))
            raise PulseError(f"Reference instruction with {key} doesn't exist in the current scope: {unassigned_keys}")
        self.references[key] = copy.deepcopy(subroutine)
    return self