from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_5a_4():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.x(0)
    qc.cx(0, 1)
    qc.x(0)
    qc.x(1)
    return qc