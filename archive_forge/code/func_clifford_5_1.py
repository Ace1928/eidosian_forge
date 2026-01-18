from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_5_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(0, 2)
    return qc