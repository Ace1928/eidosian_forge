from Bio.PopGen.GenePop import get_indiv
def start_read(self):
    """Start parsing a file containing a GenePop file."""
    self._handle = open(self.fname)
    self.comment_line = self._handle.readline().rstrip()
    sample_loci_line = self._handle.readline().rstrip().replace(',', '')
    all_loci = sample_loci_line.split(' ')
    self.loci_list.extend(all_loci)
    for line in self._handle:
        line = line.rstrip()
        if line.upper() == 'POP':
            break
        self.loci_list.append(line)
    else:
        raise ValueError('No population data found, file probably not GenePop related')
    self.current_pop = 0
    self.current_ind = 0