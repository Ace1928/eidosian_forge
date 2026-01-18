import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_gate_to_operation_to_gate_round_trips():

    def all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in all_subclasses(c)])
    gate_subclasses = {g for g in all_subclasses(cirq.Gate) if 'cirq.' in g.__module__ and 'contrib' not in g.__module__ and ('test' not in g.__module__)}
    test_module_spec = cirq.testing.json.spec_for('cirq.protocols')
    skip_classes = {cirq.ArithmeticGate, cirq.BaseDensePauliString, cirq.EigenGate, cirq.Pauli, cirq.transformers.analytical_decompositions.two_qubit_to_fsim._BGate, cirq.transformers.measurement_transformers._ConfusionChannel, cirq.transformers.measurement_transformers._ModAdd, cirq.transformers.routing.visualize_routed_circuit._SwapPrintGate, cirq.ops.raw_types._InverseCompositeGate, cirq.circuits.qasm_output.QasmTwoQubitGate, cirq.ops.MSGate, cirq.interop.quirk.QuirkQubitPermutationGate, cirq.interop.quirk.QuirkArithmeticGate}
    skipped = set()
    for gate_cls in gate_subclasses:
        filename = test_module_spec.test_data_path.joinpath(f'{gate_cls.__name__}.json')
        if pathlib.Path(filename).is_file():
            gates = cirq.read_json(filename)
        else:
            if gate_cls in skip_classes:
                skipped.add(gate_cls)
                continue
            raise AssertionError(f'{gate_cls} has no json file, please add a json file or add to the list of classes to be skipped if there is a reason this gate should not round trip to a gate via creating an operation.')
        if not isinstance(gates, collections.abc.Iterable):
            gates = [gates]
        for gate in gates:
            if gate.num_qubits():
                qudits = [cirq.LineQid(i, d) for i, d in enumerate(cirq.qid_shape(gate))]
                assert gate.on(*qudits).gate == gate
    assert skipped == skip_classes, 'A gate that was supposed to be skipped was not, please update the list of skipped gates.'