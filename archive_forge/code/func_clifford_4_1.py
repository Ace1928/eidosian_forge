from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_4_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.cx(0, 1)
    qc.swap(0, 1)
    return qc