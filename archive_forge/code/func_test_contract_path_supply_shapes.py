import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
def test_contract_path_supply_shapes():
    eq = 'ab,bc,cd'
    shps = [(2, 3), (3, 4), (4, 5)]
    contract_path(eq, *shps, shapes=True)