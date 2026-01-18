import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def calc_fst_all(self, fname):
    """Execute GenePop and gets Fst/Fis/Fit (all populations).

        Parameters:
         - fname - file name

        Returns:
         - (multiLocusFis, multiLocusFst, multiLocus Fit),
         - Iterator of tuples
           (Locus name, Fis, Fst, Fit, Qintra, Qinter)

        Will create a file called ``fname.FST``.

        This does not return the genotype frequencies.

        """
    self._run_genepop(['.FST'], [6, 1], fname)
    with open(fname + '.FST') as f:
        line = f.readline()
        while line != '':
            if line.startswith('           All:'):
                toks = [x for x in line.rstrip().split(' ') if x != '']
                try:
                    allFis = _gp_float(toks[1])
                except ValueError:
                    allFis = None
                try:
                    allFst = _gp_float(toks[2])
                except ValueError:
                    allFst = None
                try:
                    allFit = _gp_float(toks[3])
                except ValueError:
                    allFit = None
            line = f.readline()

    def proc(self):
        if hasattr(self, 'last_line'):
            line = self.last_line
            del self.last_line
        else:
            line = self.stream.readline()
        locus = None
        fis = None
        fst = None
        fit = None
        qintra = None
        qinter = None
        while line != '':
            line = line.rstrip()
            if line.startswith('  Locus:'):
                if locus is not None:
                    self.last_line = line
                    return (locus, fis, fst, fit, qintra, qinter)
                else:
                    locus = line.split(':')[1].lstrip()
            elif line.startswith('Fis^='):
                fis = _gp_float(line.split(' ')[1])
            elif line.startswith('Fst^='):
                fst = _gp_float(line.split(' ')[1])
            elif line.startswith('Fit^='):
                fit = _gp_float(line.split(' ')[1])
            elif line.startswith('1-Qintra^='):
                qintra = _gp_float(line.split(' ')[1])
            elif line.startswith('1-Qinter^='):
                qinter = _gp_float(line.split(' ')[1])
                return (locus, fis, fst, fit, qintra, qinter)
            line = self.stream.readline()
        if locus is not None:
            return (locus, fis, fst, fit, qintra, qinter)
        self.stream.close()
        self.done = True
        raise StopIteration
    return ((allFis, allFst, allFit), _FileIterator(proc, fname + '.FST'))