import pytest
import cirq
def test_default_text_diagram():

    class DiagramGate(cirq.PauliStringGateOperation):

        def map_qubits(self, qubit_map):
            pass

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
            return self._pauli_string_diagram_info(args)
    q0, q1, q2 = _make_qubits(3)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z})
    circuit = cirq.Circuit(DiagramGate(ps), DiagramGate(-ps))
    cirq.testing.assert_has_diagram(circuit, '\nq0: ───[X]───[X]───\n       │     │\nq1: ───[Y]───[Y]───\n       │     │\nq2: ───[Z]───[Z]───\n')