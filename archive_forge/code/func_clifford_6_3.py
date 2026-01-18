from qiskit.circuit.quantumcircuit import QuantumCircuit
def clifford_6_3():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.swap(0, 1)
    qc.cz(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.h(1)
    return qc