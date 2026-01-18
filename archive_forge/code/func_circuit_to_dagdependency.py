from qiskit.dagcircuit.dagdependency import DAGDependency
def circuit_to_dagdependency(circuit, create_preds_and_succs=True):
    """Build a ``DAGDependency`` object from a :class:`~.QuantumCircuit`.

    Args:
        circuit (QuantumCircuit): the input circuit.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = DAGDependency()
    dagdependency.name = circuit.name
    dagdependency.metadata = circuit.metadata
    dagdependency.add_qubits(circuit.qubits)
    dagdependency.add_clbits(circuit.clbits)
    for register in circuit.qregs:
        dagdependency.add_qreg(register)
    for register in circuit.cregs:
        dagdependency.add_creg(register)
    for instruction in circuit.data:
        dagdependency.add_op_node(instruction.operation, instruction.qubits, instruction.clbits)
    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()
    dagdependency.calibrations = circuit.calibrations
    return dagdependency