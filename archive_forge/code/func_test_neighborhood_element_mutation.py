import pytest
import numpy as np
from ase.build import fcc111
from ase.ga.slab_operators import (CutSpliceSlabCrossover,
def test_neighborhood_element_mutation(seed, cu_slab):
    rng = np.random.RandomState(seed)
    op = NeighborhoodElementMutation(element_pools=[['Cu', 'Ni', 'Au']], rng=rng)
    child, desc = op.get_new_individual([cu_slab])
    assert (child.symbols == 'Ni').sum() == 24