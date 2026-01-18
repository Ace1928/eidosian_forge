from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_6a_4():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.cx(1, 2)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.cx(2, 1)
    qc.ccx(0, 1, 2)
    qc.cx(2, 1)
    return qc