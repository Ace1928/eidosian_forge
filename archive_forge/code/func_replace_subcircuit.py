from functools import reduce
from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate
def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    """Replaces a subcircuit with another subcircuit in circuit,
    if it exists.

    Explanation
    ===========

    If multiple instances of subcircuit exists, the first instance is
    replaced.  The position to being searching from (if different from
    0) may be optionally given.  If subcircuit cannot be found, circuit
    is returned.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A quantum circuit.
    subcircuit : tuple, Gate or Mul
        The circuit to be replaced.
    replace : tuple, Gate or Mul
        The replacement circuit.
    pos : int
        The location to start search and replace
        subcircuit, if it exists.  This may be used
        if it is known beforehand that multiple
        instances exist, and it is desirable to
        replace a specific instance.  If a negative number
        is given, pos will be defaulted to 0.

    Examples
    ========

    Find and remove the subcircuit:

    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> replace_subcircuit(circuit, subcircuit)
    (X(0), H(0), X(0), H(0), Y(0))

    Remove the subcircuit given a starting search point:

    >>> replace_subcircuit(circuit, subcircuit, pos=1)
    (X(0), H(0), X(0), H(0), Y(0))

    >>> replace_subcircuit(circuit, subcircuit, pos=2)
    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))

    Replace the subcircuit:

    >>> replacement = H(0)*Z(0)
    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)
    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))
    """
    if pos < 0:
        pos = 0
    if isinstance(circuit, Mul):
        circuit = circuit.args
    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args
    if isinstance(replace, Mul):
        replace = replace.args
    elif replace is None:
        replace = ()
    loc = find_subcircuit(circuit, subcircuit, start=pos)
    if loc > -1:
        left = circuit[0:loc]
        right = circuit[loc + len(subcircuit):len(circuit)]
        circuit = left + replace + right
    return circuit