from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def batch_vjp(self, tapes, grad_vecs, reduction='append', starting_state=None, use_device_state=False):
    """Generate the processing function required to compute the vector-Jacobian products
        of a batch of tapes.

        Args:
            tapes (Sequence[.QuantumTape]): sequence of quantum tapes to differentiate
            grad_vecs (Sequence[tensor_like]): Sequence of gradient-output vectors ``grad_vec``.
                Must be the same length as ``tapes``. Each ``grad_vec`` tensor should have
                shape matching the output shape of the corresponding tape.
            reduction (str): Determines how the vector-Jacobian products are returned.
                If ``append``, then the output of the function will be of the form
                ``List[tensor_like]``, with each element corresponding to the VJP of each
                input tape. If ``extend``, then the output VJPs will be concatenated.
            starting_state (tensor_like): post-forward pass state to start execution with.
                It should be complex-valued. Takes precedence over ``use_device_state``.
            use_device_state (bool): use current device state to initialize. A forward pass of
                the same circuit should be the last thing the device has executed.
                If a ``starting_state`` is provided, that takes precedence.

        Returns:
            The processing function required to compute the vector-Jacobian products
            of a batch of tapes.
        """
    fns = []
    for tape, grad_vec in zip(tapes, grad_vecs):
        fun = self.vjp(tape.measurements, grad_vec, starting_state=starting_state, use_device_state=use_device_state)
        fns.append(fun)

    def processing_fns(tapes):
        vjps = []
        for tape, fun in zip(tapes, fns):
            vjp = fun(tape)
            if not isinstance(vjp, tuple) and getattr(reduction, '__name__', reduction) == 'extend':
                vjp = (vjp,)
            if isinstance(reduction, str):
                getattr(vjps, reduction)(vjp)
            elif callable(reduction):
                reduction(vjps, vjp)
        return vjps
    return processing_fns