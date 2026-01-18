import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def ld_func(self):
    line = self.stream.readline().rstrip()
    if line == '':
        self.done = True
        raise StopIteration
    toks = [x for x in line.split(' ') if x != '']
    locus1, locus2 = (toks[0], toks[2])
    try:
        chi2, df, p = (_gp_float(toks[3]), _gp_int(toks[4]), _gp_float(toks[5]))
    except ValueError:
        return ((locus1, locus2), None)
    return ((locus1, locus2), (chi2, df, p))