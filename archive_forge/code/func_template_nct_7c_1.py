from qiskit.circuit.quantumcircuit import QuantumCircuit
def template_nct_7c_1():
    """
    Returns:
        QuantumCircuit: template as a quantum circuit.
    """
    qc = QuantumCircuit(3)
    qc.x(0)
    qc.ccx(0, 2, 1)
    qc.cx(1, 2)
    qc.ccx(0, 1, 2)
    qc.ccx(0, 2, 1)
    qc.x(0)
    qc.ccx(0, 1, 2)
    return qc