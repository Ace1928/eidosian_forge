from typing import Callable, Sequence, Tuple
import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra
from cirq_ft.algos import and_gate, unary_iteration_gate
from numpy.typing import ArrayLike, NDArray
def decompose_zero_selection(self, context: cirq.DecompositionContext, **quregs: NDArray[cirq.Qid]) -> cirq.OP_TREE:
    controls = infra.merge_qubits(self.control_registers, **quregs)
    target_regs = {reg.name: quregs[reg.name] for reg in self.target_registers}
    zero_indx = (0,) * len(self.data[0].shape)
    if self.num_controls == 0:
        yield self._load_nth_data(zero_indx, cirq.X, **target_regs)
    elif self.num_controls == 1:
        yield self._load_nth_data(zero_indx, lambda q: cirq.CNOT(controls[0], q), **target_regs)
    else:
        and_ancilla = context.qubit_manager.qalloc(len(controls) - 2)
        and_target = context.qubit_manager.qalloc(1)[0]
        multi_controlled_and = and_gate.And((1,) * len(controls)).on_registers(ctrl=np.array(controls)[:, np.newaxis], junk=np.array(and_ancilla)[:, np.newaxis], target=and_target)
        yield multi_controlled_and
        yield self._load_nth_data(zero_indx, lambda q: cirq.CNOT(and_target, q), **target_regs)
        yield cirq.inverse(multi_controlled_and)
        context.qubit_manager.qfree(and_ancilla + [and_target])