import os
import re
import shutil
import tempfile
from Bio.Application import AbstractCommandline, _Argument
def pop_parser(self):
    if hasattr(self, 'old_line'):
        line = self.old_line
        del self.old_line
    else:
        line = self.stream.readline()
    loci_content = {}
    while line != '':
        line = line.rstrip()
        if 'Tables of allelic frequencies for each locus' in line:
            return (self.curr_pop, loci_content)
        match = re.match('.*Pop: (.+) Locus: (.+)', line)
        if match is not None:
            pop = match.group(1).rstrip()
            locus = match.group(2)
            if not hasattr(self, 'first_locus'):
                self.first_locus = locus
            if hasattr(self, 'curr_pop'):
                if self.first_locus == locus:
                    old_pop = self.curr_pop
                    self.old_line = line
                    del self.first_locus
                    del self.curr_pop
                    return (old_pop, loci_content)
            self.curr_pop = pop
        else:
            line = self.stream.readline()
            continue
        geno_list = []
        line = self.stream.readline()
        if 'No data' in line:
            continue
        while 'Genotypes  Obs.' not in line:
            line = self.stream.readline()
        while line != '\n':
            m2 = re.match(' +([0-9]+) , ([0-9]+) *([0-9]+) *(.+)', line)
            if m2 is not None:
                geno_list.append((_gp_int(m2.group(1)), _gp_int(m2.group(2)), _gp_int(m2.group(3)), _gp_float(m2.group(4))))
            else:
                line = self.stream.readline()
                continue
            line = self.stream.readline()
        while 'Expected number of ho' not in line:
            line = self.stream.readline()
        expHo = _gp_float(line[38:])
        line = self.stream.readline()
        obsHo = _gp_int(line[38:])
        line = self.stream.readline()
        expHe = _gp_float(line[38:])
        line = self.stream.readline()
        obsHe = _gp_int(line[38:])
        line = self.stream.readline()
        while 'Sample count' not in line:
            line = self.stream.readline()
        line = self.stream.readline()
        freq_fis = {}
        overall_fis = None
        while '----' not in line:
            vals = [x for x in line.rstrip().split(' ') if x != '']
            if vals[0] == 'Tot':
                overall_fis = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
            else:
                freq_fis[_gp_int(vals[0])] = (_gp_int(vals[1]), _gp_float(vals[2]), _gp_float(vals[3]))
            line = self.stream.readline()
        loci_content[locus] = (geno_list, (expHo, obsHo, expHe, obsHe), freq_fis, overall_fis)
    self.done = True
    raise StopIteration