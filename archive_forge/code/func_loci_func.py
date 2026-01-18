import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def loci_func(self):
    line = self.stream.readline()
    while line != '':
        line = line.rstrip()
        m = re.search(' Locus: (.+)', line)
        if m is not None:
            locus = m.group(1)
            matrix = _read_headed_triangle_matrix(self.stream)
            return (locus, matrix)
        line = self.stream.readline()
    self.done = True
    raise StopIteration