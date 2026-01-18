from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_6_4():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    qc.s(0)
    qc.h(0)
    return qc