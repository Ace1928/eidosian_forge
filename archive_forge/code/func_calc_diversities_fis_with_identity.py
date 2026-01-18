import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def calc_diversities_fis_with_identity(self, fname):
    """Compute identity-base Gene diversities and Fis."""
    return self._calc_diversities_fis(fname, '.DIV')