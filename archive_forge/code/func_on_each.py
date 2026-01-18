import abc
import functools
from typing import (
from typing_extensions import Self
import numpy as np
import sympy
from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv
def on_each(self, *targets: Union[Qid, Iterable[Any]]) -> List['cirq.Operation']:
    """Returns a list of operations applying the gate to all targets.

        Args:
            *targets: The qubits to apply this gate to. For single-qubit gates
                this can be provided as varargs or a combination of nested
                iterables. For multi-qubit gates this must be provided as an
                `Iterable[Sequence[Qid]]`, where each sequence has `num_qubits`
                qubits.

        Returns:
            Operations applying this gate to the target qubits.

        Raises:
            ValueError: If targets are not instances of Qid or Iterable[Qid].
                If the gate qubit number is incompatible.
            TypeError: If a single target is supplied and it is not iterable.
        """
    operations: List['cirq.Operation'] = []
    if self._num_qubits_() > 1:
        iterator: Iterable = targets
        if len(targets) == 1:
            if not isinstance(targets[0], Iterable):
                raise TypeError(f'{targets[0]} object is not iterable.')
            t0 = list(targets[0])
            iterator = [t0] if t0 and isinstance(t0[0], Qid) else t0
        if __cirq_debug__.get():
            for target in iterator:
                if not isinstance(target, Sequence):
                    raise ValueError(f'Inputs to multi-qubit gates must be Sequence[Qid]. Type: {type(target)}')
                if not all((isinstance(x, Qid) for x in target)):
                    raise ValueError(f'All values in sequence should be Qids, but got {target}')
                if len(target) != self._num_qubits_():
                    raise ValueError(f'Expected {self._num_qubits_()} qubits, got {target}')
                operations.append(self.on(*target))
        else:
            operations = [self.on(*target) for target in iterator]
        return operations
    if not __cirq_debug__.get():
        return [op for q in targets for op in (self.on_each(*q) if isinstance(q, Iterable) and (not isinstance(q, str)) else [self.on(cast('cirq.Qid', q))])]
    for target in targets:
        if isinstance(target, Qid):
            operations.append(self.on(target))
        elif isinstance(target, Iterable) and (not isinstance(target, str)):
            operations.extend(self.on_each(*target))
        else:
            raise ValueError(f'Gate was called with type different than Qid. Type: {type(target)}')
    return operations