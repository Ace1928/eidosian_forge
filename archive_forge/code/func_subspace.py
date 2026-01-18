import numpy as np
from pennylane.operation import Operation, AdjointUndefinedError
from pennylane.wires import Wires
from .parametric_ops import validate_subspace
@property
def subspace(self):
    """The single-qutrit basis states which the operator acts on

        This property returns the 2D subspace on which the operator acts if specified,
        or None if no subspace is defined. This subspace determines which two single-qutrit
        basis states the operator acts on. The remaining basis state is not affected by the
        operator.

        Returns:
            tuple[int] or None: subspace on which operator acts, if specified, else None
        """
    return self._subspace