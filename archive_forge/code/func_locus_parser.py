import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def locus_parser(self):
    line = self.stream.readline()
    while line != '':
        line = line.rstrip()
        match = re.match(' Locus: (.+)', line)
        if match is not None:
            locus = match.group(1)
            alleles, table = _read_allele_freq_table(self.stream)
            return (locus, alleles, table)
        line = self.stream.readline()
    self.done = True
    raise StopIteration