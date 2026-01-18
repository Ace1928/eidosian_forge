import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def test_global_hz_excess(self, fname, enum_test=True, dememorization=10000, batches=20, iterations=5000):
    """Use Global Hardy-Weinberg test for heterozygote excess.

        Returns a triple with:
         - A list per population containing (pop_name, P-val, SE, switches).
           Some pops have a None if the info is not available.
           SE might be none (for enumerations).
         - A list per loci containing (locus_name, P-val, SE, switches).
           Some loci have a None if the info is not available.
           SE might be none (for enumerations).
         - Overall results (P-val, SE, switches)

        """
    return self._test_global_hz_both(fname, 5, '.EG', enum_test, dememorization, batches, iterations)