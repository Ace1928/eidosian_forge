from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_4a_2():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(4)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    qc.ccx(0, 1, 3)
    qc.cx(1, 2)
    return qc