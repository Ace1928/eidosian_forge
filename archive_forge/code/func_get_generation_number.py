import os
from ase import Atoms
from ase.ga import get_raw_score
from ase.ga import set_parametrization, set_neighbor_list
import ase.db
def get_generation_number(self, size=None):
    """ Returns the current generation number, by looking
            at the number of relaxed individuals and comparing
            this number to the supplied size or population size.

            If all individuals in generation 3 has been relaxed
            it will return 4 if not all in generation 4 has been
            relaxed.
        """
    if size is None:
        size = self.get_param('population_size')
    if size is None:
        return 0
    lg = size
    g = 0
    all_candidates = list(self.c.select(relaxed=1))
    while lg > 0:
        lg = len([c for c in all_candidates if c.generation == g])
        if lg >= size:
            g += 1
        else:
            return g