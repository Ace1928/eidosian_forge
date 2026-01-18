from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
def bfs_identity_search(gate_list, nqubits, max_depth=None, identity_only=False):
    """Constructs a set of gate identities from the list of possible gates.

    Performs a breadth first search over the space of gate identities.
    This allows the finding of the shortest gate identities first.

    Parameters
    ==========

    gate_list : list, Gate
        A list of Gates from which to search for gate identities.
    nqubits : int
        The number of qubits the quantum circuit operates on.
    max_depth : int
        The longest quantum circuit to construct from gate_list.
    identity_only : bool
        True to search for gate identities that reduce to identity;
        False to search for gate identities that reduce to a scalar.

    Examples
    ========

    Find a list of gate identities:

    >>> from sympy.physics.quantum.identitysearch import bfs_identity_search
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> bfs_identity_search([x], 1, max_depth=2)
    {GateIdentity(X(0), X(0))}

    >>> bfs_identity_search([x, y, z], 1)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0)), GateIdentity(X(0), Y(0), Z(0))}

    Find a list of identities that only equal to 1:

    >>> bfs_identity_search([x, y, z], 1, identity_only=True)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0))}
    """
    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)
    id_only = identity_only
    queue = deque([()])
    ids = set()
    while len(queue) > 0:
        current_circuit = queue.popleft()
        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)
            circuit_reducible = is_reducible(new_circuit, nqubits, 1, len(new_circuit))
            if is_scalar_matrix(new_circuit, nqubits, id_only) and (not is_degenerate(ids, new_circuit)) and (not circuit_reducible):
                ids.add(GateIdentity(*new_circuit))
            elif len(new_circuit) < max_depth and (not circuit_reducible):
                queue.append(new_circuit)
    return ids