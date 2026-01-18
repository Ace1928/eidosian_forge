from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def stochiometry(self):
    """Compute the stoichiometry matrix for self.

        Returns (species, reactions, stoch) where:
         - species    = ordered list of species in this system
         - reactions  = ordered list of reactions in this system
         - stoch      = 2D array where stoch[i][j] is coef of the
           jth species in the ith reaction, as defined
           by species and reactions above

        """
    species = self.species()
    reactions = self.reactions()
    stoch = [] * len(reactions)
    for i in range(len(reactions)):
        stoch[i] = 0 * len(species)
        for s in reactions[i].species():
            stoch[species.index(s)] = reactions[i].reactants[s]
    return (species, reactions, stoch)