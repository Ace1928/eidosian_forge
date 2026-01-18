import math
import warnings
from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.ops import Adjoint
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition
@transform
def clifford_t_decomposition(tape: QuantumTape, epsilon=0.0001, max_expansion=6, method='sk', **method_kwargs) -> (Sequence[QuantumTape], Callable):
    """Decomposes a circuit into the Clifford+T basis.

    This method first decomposes the gate operations to a basis comprised of Clifford, :class:`~.T`, :class:`~.RZ` and
    :class:`~.GlobalPhase` operations (and their adjoints). The Clifford gates include the following PennyLane operations:

    - Single qubit gates - :class:`~.Identity`, :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`,
      :class:`~.SX`, :class:`~.S`, and :class:`~.Hadamard`.
    - Two qubit gates - :class:`~.CNOT`, :class:`~.CY`, :class:`~.CZ`, :class:`~.SWAP`, and :class:`~.ISWAP`.

    Then, the leftover single qubit :class:`~.RZ` operations are approximated in the Clifford+T basis with
    :math:`\\epsilon > 0` error. By default, we use the Solovay-Kitaev algorithm described in
    `Dawson and Nielsen (2005) <https://arxiv.org/abs/quant-ph/0505030>`_ for this.

    Args:
        tape (QNode or QuantumTape or Callable): The quantum circuit to be decomposed.
        epsilon (float): The maximum permissible operator norm error of the complete circuit decomposition. Defaults to ``0.0001``.
        max_expansion (int): The depth to be used for tape expansion before manual decomposition to Clifford+T basis is applied.
        method (str): Method to be used for Clifford+T decomposition. Default value is ``"sk"`` for Solovay-Kitaev.
        **method_kwargs: Keyword argument to pass options for the ``method`` used for decompositions.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described
        in the :func:`qml.transform <pennylane.transform>`.

    **Keyword Arguments**

    - Solovay-Kitaev decomposition --
        **max_depth** (int), **basis_set** (list[str]), **basis_length** (int) -- arguments for the ``"sk"`` method,
        where the decomposition is performed using the :func:`~.sk_decomposition` method.

    Raises:
        ValueError: If a gate operation does not have a decomposition when required.
        NotImplementedError: If chosen decomposition ``method`` is not supported.

    .. seealso:: :func:`~.sk_decomposition` for Solovay-Kitaev decomposition.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        x, y = 1.1, 2.2
        decomposed_circuit = qml.transforms.clifford_t_decomposition(circuit)
        result = circuit(x, y)
        approx = decomposed_circuit(x, y)

    >>> qml.math.allclose(result, approx, atol=1e-4)
    True
    """
    with QueuingManager.stop_recording():
        basis_set = [op.__name__ for op in _PARAMETER_GATES + _CLIFFORD_T_GATES]
        pipelines = [remove_barrier, commute_controlled, cancel_inverses, merge_rotations]
        [compiled_tape], _ = qml.compile(tape, pipelines, basis_set=basis_set, expand_depth=max_expansion)
        decomp_ops, gphase_ops = ([], [])
        for op in compiled_tape.operations:
            if isinstance(op, _SKIP_OP_TYPES):
                decomp_ops.append(op)
            elif isinstance(op, qml.GlobalPhase):
                gphase_ops.append(op)
            elif op.name in basis_set and check_clifford_t(op):
                if op.num_params:
                    decomp_ops.extend(_rot_decompose(op))
                else:
                    decomp_ops.append(op)
            elif op.num_wires == 1:
                if op.name in basis_set:
                    d_ops = _rot_decompose(op)
                else:
                    d_ops, g_op = _one_qubit_decompose(op)
                    gphase_ops.append(g_op)
                decomp_ops.extend(d_ops)
            elif op.num_wires == 2:
                d_ops = _two_qubit_decompose(op)
                decomp_ops.extend(d_ops)
            else:
                try:
                    md_ops = op.decomposition()
                    idx = 0
                    while idx < len(md_ops):
                        md_op = md_ops[idx]
                        if md_op.name not in basis_set or not check_clifford_t(md_op):
                            if len(md_op.wires) == 1:
                                if md_op.name in basis_set:
                                    d_ops = _rot_decompose(md_op)
                                else:
                                    d_ops, g_op = _one_qubit_decompose(md_op)
                                    gphase_ops.append(g_op)
                            elif len(md_op.wires) == 2:
                                d_ops = _two_qubit_decompose(md_op)
                            else:
                                d_ops = md_op.decomposition()
                            del md_ops[idx]
                            md_ops[idx:idx] = d_ops
                        idx += 1
                    decomp_ops.extend(md_ops)
                except Exception as exc:
                    raise ValueError(f'Cannot unroll {op} into the Clifford+T basis as no rule exists for its decomposition') from exc
        merged_ops, number_ops = _merge_param_gates(decomp_ops, merge_ops=['RZ'])
        new_operations = _fuse_global_phases(merged_ops + gphase_ops)
        epsilon /= number_ops or 1
        if method == 'sk':
            decompose_fn = sk_decomposition
        else:
            raise NotImplementedError(f"Currently we only support Solovay-Kitaev ('sk') decompostion, got {method}")
        decomp_ops = []
        phase = new_operations.pop().data[0]
        for op in new_operations:
            if isinstance(op, qml.RZ):
                clifford_ops = decompose_fn(op, epsilon, **method_kwargs)
                phase += qml.math.convert_like(clifford_ops.pop().data[0], phase)
                decomp_ops.extend(clifford_ops)
            else:
                decomp_ops.append(op)
        if qml.math.is_abstract(phase) or not qml.math.allclose(phase, 0.0):
            decomp_ops.append(qml.GlobalPhase(phase))
    new_tape = type(tape)(decomp_ops, compiled_tape.measurements, shots=tape.shots)
    [new_tape], _ = cancel_inverses(new_tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]
    return ([new_tape], null_postprocessing)