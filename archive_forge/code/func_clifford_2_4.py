from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_2_4():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.swap(1, 0)
    return qc