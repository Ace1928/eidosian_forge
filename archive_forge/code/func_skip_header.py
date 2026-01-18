from Bio.PopGen.GenePop import get_indiv
def skip_header(self):
    """Skip the Header. To be done after a re-open."""
    self.current_pop = 0
    self.current_ind = 0
    for line in self._handle:
        if line.rstrip().upper() == 'POP':
            return