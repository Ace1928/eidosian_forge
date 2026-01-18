import pytest
import numpy as np
from ase.build import fcc111
from ase.ga.slab_operators import (CutSpliceSlabCrossover,
def test_cut_splice(seed, cu_slab):
    rng = np.random.RandomState(seed)
    ratio = 0.4
    op = CutSpliceSlabCrossover(min_ratio=ratio, rng=rng)
    p1 = cu_slab
    natoms = len(p1)
    p2 = cu_slab.copy()
    p2.symbols = ['Au'] * natoms
    p2.info['confid'] = 2
    child, desc = op.get_new_individual([p1, p2])
    assert desc == 'CutSpliceSlabCrossover: Parents 1 2'
    syms = child.get_chemical_symbols()
    new_ratio = syms.count('Au') / natoms
    assert new_ratio > ratio and new_ratio < 1 - ratio
    op = CutSpliceSlabCrossover(element_pools=['Cu', 'Au'], allowed_compositions=[(12, 12)], rng=rng)
    child = op.operate(p1, p2)
    assert child.get_chemical_symbols().count('Au') == 12