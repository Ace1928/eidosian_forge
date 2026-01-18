from .Controller import GenePopController
from Bio.PopGen import GenePop
def test_hw_global(self, test_type='deficiency', enum_test=True, dememorization=10000, batches=20, iterations=5000):
    """Perform Hardy-Weinberg global Heterozygote test."""
    if test_type == 'deficiency':
        pop_res, loc_res, all = self._controller.test_global_hz_deficiency(self._fname, enum_test, dememorization, batches, iterations)
    else:
        pop_res, loc_res, all = self._controller.test_global_hz_excess(self._fname, enum_test, dememorization, batches, iterations)
    return (list(pop_res), list(loc_res), all)