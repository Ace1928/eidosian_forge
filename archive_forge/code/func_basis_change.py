import abc
from typing import (
from cirq import ops, value, devices
def basis_change(self) -> 'cirq.OP_TREE':
    return self._basis_change