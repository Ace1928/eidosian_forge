from sympy.core.relational import Ne
from sympy.core.symbol import (Dummy, Symbol, symbols)
from sympy.functions.elementary.complexes import (adjoint, conjugate, transpose)
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.tensor_functions import (Eijk, KroneckerDelta, LeviCivita)
from sympy.physics.secondquant import evaluate_deltas, F
def test_levicivita():
    assert Eijk(1, 2, 3) == LeviCivita(1, 2, 3)
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(int(1), int(2), int(3)) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    i, j, k = symbols('i j k')
    assert LeviCivita(i, j, k) == LeviCivita(i, j, k, evaluate=False)
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(i, j, k).doit() == (j - i) * (k - i) * (k - j) / 2
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1
    assert LeviCivita(i, j, k).is_integer is True
    assert adjoint(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert conjugate(LeviCivita(i, j, k)) == LeviCivita(i, j, k)
    assert transpose(LeviCivita(i, j, k)) == LeviCivita(i, j, k)