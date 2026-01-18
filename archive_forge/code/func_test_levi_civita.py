from symengine import (
from symengine.test_utilities import raises
import unittest
def test_levi_civita():
    i = Symbol('i')
    j = Symbol('j')
    assert LeviCivita(1, 2, 3) == 1
    assert LeviCivita(1, 3, 2) == -1
    assert LeviCivita(1, 2, 2) == 0
    assert LeviCivita(i, j, i) == 0
    assert LeviCivita(1, i, i) == 0
    assert LeviCivita(1, 2, 3, 1) == 0
    assert LeviCivita(4, 5, 1, 2, 3) == 1
    assert LeviCivita(4, 5, 2, 1, 3) == -1