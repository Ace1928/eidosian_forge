import cirq
import cirq.contrib.qcircuit as ccq
import cirq.testing as ct
def test_fallback_diagram():

    class MagicGate(cirq.testing.ThreeQubitGate):

        def __str__(self):
            return 'MagicGate'

    class MagicOp(cirq.Operation):

        def __init__(self, *qubits):
            self._qubits = qubits

        def with_qubits(self, *new_qubits):
            return MagicOp(*new_qubits)

        @property
        def qubits(self):
            return self._qubits

        def __str__(self):
            return 'MagicOperate'
    circuit = cirq.Circuit(MagicOp(cirq.NamedQubit('b')), MagicGate().on(cirq.NamedQubit('b'), cirq.NamedQubit('a'), cirq.NamedQubit('c')))
    expected_diagram = '\n\\Qcircuit @R=1em @C=0.75em {\n \\\\\n &\\lstick{\\text{a}}& \\qw&                           \\qw&\\gate{\\text{\\#2}}       \\qw    &\\qw\\\\\n &\\lstick{\\text{b}}& \\qw&\\gate{\\text{MagicOperate}} \\qw&\\gate{\\text{MagicGate}} \\qw\\qwx&\\qw\\\\\n &\\lstick{\\text{c}}& \\qw&                           \\qw&\\gate{\\text{\\#3}}       \\qw\\qwx&\\qw\\\\\n \\\\\n}'.strip()
    assert_has_qcircuit_diagram(circuit, expected_diagram)