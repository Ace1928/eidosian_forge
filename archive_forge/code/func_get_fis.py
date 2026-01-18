from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_fis(self, pop_pos, locus_name):
    """Return the Fis for a certain population and locus.

        Below CW means Cockerham and Weir and RH means Robertson and Hill.

        Returns a pair:

        - dictionary [allele] = (repetition count, frequency, Fis CW )
          with information for each allele
        - a triple with total number of alleles, Fis CW, Fis RH

        """
    geno_freqs = self._controller.calc_allele_genotype_freqs(self._fname)
    pop_iter, loc_iter = geno_freqs
    pops = list(pop_iter)
    return pops[pop_pos][1][locus_name][2:]