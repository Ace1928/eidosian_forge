from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_2a_2():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.cx(0, 1)
    return qc