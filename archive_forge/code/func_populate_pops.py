from ase.ga import get_raw_score
def populate_pops(self, to_gen):
    """Populate the pops dictionary with how the population
        looked after i number of generations."""
    for i in range(to_gen):
        if i not in self.pops.keys():
            self.pops[i] = self.pop.get_population_after_generation(i)