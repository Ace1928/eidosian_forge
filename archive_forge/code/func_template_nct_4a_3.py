from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_4a_3():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.cx(0, 2)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 1)
    return qc