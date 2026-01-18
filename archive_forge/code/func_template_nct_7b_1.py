from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_7b_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.cx(0, 1)
    qc.x(0)
    qc.ccx(0, 1, 2)
    return qc