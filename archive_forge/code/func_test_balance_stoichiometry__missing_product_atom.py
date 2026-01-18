from functools import reduce
from operator import attrgetter, add
import sys
from sympy import nsimplify
import pytest
from ..util.arithmeticdict import ArithmeticDict
from ..util.testing import requires
from ..util.parsing import parsing_library
from ..units import default_units, units_library, to_unitless, allclose
from ..chemistry import (
@requires('sympy')
def test_balance_stoichiometry__missing_product_atom():
    with pytest.raises(ValueError):
        balance_stoichiometry({'C7H5(NO2)3', 'Al', 'NH4NO3'}, {'CO', 'H2O', 'N2'})