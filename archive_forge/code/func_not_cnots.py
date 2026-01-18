import pytest
import cirq
import cirq.contrib.graph_device as ccgd
import cirq.contrib.graph_device.graph_device as ccgdgd
def not_cnots(first_op, second_op):
    if all((isinstance(op, cirq.GateOperation) and op.gate == cirq.CNOT for op in (first_op, second_op))):
        raise ValueError('Simultaneous CNOTs')