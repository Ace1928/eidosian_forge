from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_6_2():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.s(0)
    qc.s(1)
    qc.cx(0, 1)
    qc.sdg(1)
    qc.cx(0, 1)
    qc.cz(0, 1)
    return qc