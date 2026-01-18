from Bio.PopGen.GenePop import get_indiv
def skip_population(self):
    """Skip the current population. Returns true if there is another pop."""
    for line in self._handle:
        if line == '':
            return False
        line = line.rstrip()
        if line.upper() == 'POP':
            self.current_pop += 1
            self.current_ind = 0
            return True