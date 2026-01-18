from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_4b_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(4)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 2, 3)
    qc.ccx(1, 2, 3)
    qc.ccx(0, 2, 3)
    return qc