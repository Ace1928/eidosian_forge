from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_3_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(1)
    qc.s(0)
    qc.s(0)
    qc.z(0)
    return qc