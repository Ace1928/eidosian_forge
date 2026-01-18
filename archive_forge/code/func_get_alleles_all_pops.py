from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_alleles_all_pops(self, locus_name):
    """Return the alleles for a certain population and locus."""
    geno_freqs = self._controller.calc_allele_genotype_freqs(self._fname)
    pop_iter, loc_iter = geno_freqs
    for locus_info in loc_iter:
        if locus_info[0] == locus_name:
            return locus_info[1]