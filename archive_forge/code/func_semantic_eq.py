from qiskit.exceptions import QiskitError
@staticmethod
def semantic_eq(node1, node2):
    """
        Check if DAG nodes are considered equivalent, e.g., as a node_match for nx.is_isomorphic.

        Args:
            node1 (DAGDepNode): A node to compare.
            node2 (DAGDepNode): The other node to compare.

        Return:
            Bool: If node1 == node2
        """
    if 'barrier' == node1.name == node2.name:
        return set(node1._qargs) == set(node2._qargs)
    if node1.type == node2.type:
        if node1._op == node2._op:
            if node1.name == node2.name:
                if node1._qargs == node2._qargs:
                    if node1.cargs == node2.cargs:
                        if node1.type == 'op':
                            if getattr(node1._op, 'condition', None) != getattr(node2._op, 'condition', None):
                                return False
                        return True
    return False