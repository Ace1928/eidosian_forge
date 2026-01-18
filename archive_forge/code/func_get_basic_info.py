from .Controller import GenePopController
from Bio.PopGen import GenePop
def get_basic_info(self):
    """Obtain the population list and loci list from the file."""
    with open(self._fname) as f:
        rec = GenePop.read(f)
    return (rec.pop_list, rec.loci_list)