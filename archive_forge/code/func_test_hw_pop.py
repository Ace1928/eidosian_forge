from .Controller import GenePopController
from Bio.PopGen import GenePop
def test_hw_pop(self, pop_pos, test_type='probability'):
    """Perform Hardy-Weinberg test on the given position."""
    if test_type == 'deficiency':
        hw_res = self._controller.test_pop_hz_deficiency(self._fname)
    elif test_type == 'excess':
        hw_res = self._controller.test_pop_hz_excess(self._fname)
    else:
        loci_res, hw_res, fisher_full = self._controller.test_pop_hz_prob(self._fname, '.P')
    for i in range(pop_pos - 1):
        next(hw_res)
    return next(hw_res)