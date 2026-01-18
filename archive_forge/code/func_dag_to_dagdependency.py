from qiskit.dagcircuit.dagdependency import DAGDependency
def dag_to_dagdependency(dag, create_preds_and_succs=True):
    """Build a ``DAGDependency`` object from a ``DAGCircuit``.

    Args:
        dag (DAGCircuit): the input dag.
        create_preds_and_succs (bool): whether to construct lists of
            predecessors and successors for every node.

    Return:
        DAGDependency: the DAG representing the input circuit as a dag dependency.
    """
    dagdependency = DAGDependency()
    dagdependency.name = dag.name
    dagdependency.metadata = dag.metadata
    dagdependency.add_qubits(dag.qubits)
    dagdependency.add_clbits(dag.clbits)
    for register in dag.qregs.values():
        dagdependency.add_qreg(register)
    for register in dag.cregs.values():
        dagdependency.add_creg(register)
    for node in dag.topological_op_nodes():
        inst = node.op.copy()
        dagdependency.add_op_node(inst, node.qargs, node.cargs)
    if create_preds_and_succs:
        dagdependency._add_predecessors()
        dagdependency._add_successors()
    dagdependency.global_phase = dag.global_phase
    dagdependency.calibrations = dag.calibrations
    return dagdependency