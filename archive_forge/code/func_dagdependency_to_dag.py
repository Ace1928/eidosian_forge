from qiskit.dagcircuit.dagcircuit import DAGCircuit
def dagdependency_to_dag(dagdependency):
    """Build a ``DAGCircuit`` object from a ``DAGDependency``.

    Args:
        dag dependency (DAGDependency): the input dag.

    Return:
        DAGCircuit: the DAG representing the input circuit.
    """
    dagcircuit = DAGCircuit()
    dagcircuit.name = dagdependency.name
    dagcircuit.metadata = dagdependency.metadata
    dagcircuit.add_qubits(dagdependency.qubits)
    dagcircuit.add_clbits(dagdependency.clbits)
    for register in dagdependency.qregs.values():
        dagcircuit.add_qreg(register)
    for register in dagdependency.cregs.values():
        dagcircuit.add_creg(register)
    for node in dagdependency.topological_nodes():
        inst = node.op.copy()
        dagcircuit.apply_operation_back(inst, node.qargs, node.cargs)
    dagcircuit.global_phase = dagdependency.global_phase
    dagcircuit.calibrations = dagdependency.calibrations
    return dagcircuit