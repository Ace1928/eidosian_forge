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
def generate_gate_rules(gate_seq, return_as_muls=False):
    """Returns a set of gate rules.  Each gate rules is represented
    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary
    scalar value.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules.

    A gate rule is an expression such as ABC = D or AB = CD, where
    A, B, C, and D are gates.  Each value on either side of the
    equal sign represents a circuit.  The four operations allow
    one to find a set of equivalent circuits from a gate identity.
    The letters denoting the operation tell the user what
    activities to perform on each expression.  The first letter
    indicates which side of the equal sign to focus on.  The
    second letter indicates which gate to focus on given the
    side.  Once this information is determined, the inverse
    of the gate is multiplied on both circuits to create a new
    gate rule.

    For example, given the identity, ABCD = 1, a LL operation
    means look at the left value and multiply both left sides by the
    inverse of the leftmost gate A.  If A is Hermitian, the inverse
    of A is still A.  The resulting new rule is BCD = A.

    The following is a summary of the four operations.  Assume
    that in the examples, all gates are Hermitian.

        LL : left circuit, left multiply
             ABCD = E -> AABCD = AE -> BCD = AE
        LR : left circuit, right multiply
             ABCD = E -> ABCDD = ED -> ABC = ED
        RL : right circuit, left multiply
             ABC = ED -> EABC = EED -> EABC = D
        RR : right circuit, right multiply
             AB = CD -> ABD = CDD -> ABD = C

    The number of gate rules generated is n*(n+1), where n
    is the number of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix
    return_as_muls : bool
        True to return a set of Muls; False to return a set of tuples

    Examples
    ========

    Find the gate rules of the current circuit using tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_gate_rules
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_gate_rules((x, x))
    {((X(0),), (X(0),)), ((X(0), X(0)), ())}

    >>> generate_gate_rules((x, y, z))
    {((), (X(0), Z(0), Y(0))), ((), (Y(0), X(0), Z(0))),
     ((), (Z(0), Y(0), X(0))), ((X(0),), (Z(0), Y(0))),
     ((Y(0),), (X(0), Z(0))), ((Z(0),), (Y(0), X(0))),
     ((X(0), Y(0)), (Z(0),)), ((Y(0), Z(0)), (X(0),)),
     ((Z(0), X(0)), (Y(0),)), ((X(0), Y(0), Z(0)), ()),
     ((Y(0), Z(0), X(0)), ()), ((Z(0), X(0), Y(0)), ())}

    Find the gate rules of the current circuit using Muls:

    >>> generate_gate_rules(x*x, return_as_muls=True)
    {(1, 1)}

    >>> generate_gate_rules(x*y*z, return_as_muls=True)
    {(1, X(0)*Z(0)*Y(0)), (1, Y(0)*X(0)*Z(0)),
     (1, Z(0)*Y(0)*X(0)), (X(0)*Y(0), Z(0)),
     (Y(0)*Z(0), X(0)), (Z(0)*X(0), Y(0)),
     (X(0)*Y(0)*Z(0), 1), (Y(0)*Z(0)*X(0), 1),
     (Z(0)*X(0)*Y(0), 1), (X(0), Z(0)*Y(0)),
     (Y(0), X(0)*Z(0)), (Z(0), Y(0)*X(0))}
    """
    if isinstance(gate_seq, Number):
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            return {((), ())}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args
    queue = deque()
    rules = set()
    max_ops = len(gate_seq)

    def process_new_rule(new_rule, ops):
        if new_rule is not None:
            new_left, new_right = new_rule
            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)
            if ops + 1 < max_ops:
                queue.append(new_rule + (ops + 1,))
    queue.append((gate_seq, (), 0))
    rules.add((gate_seq, ()))
    while len(queue) > 0:
        left, right, ops = queue.popleft()
        new_rule = ll_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = lr_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = rl_op(left, right)
        process_new_rule(new_rule, ops)
        new_rule = rr_op(left, right)
        process_new_rule(new_rule, ops)
    if return_as_muls:
        mul_rules = set()
        for rule in rules:
            left, right = rule
            mul_rules.add((Mul(*left), Mul(*right)))
        rules = mul_rules
    return rules