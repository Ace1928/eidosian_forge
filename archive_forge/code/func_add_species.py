from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def add_species(self, species):
    """Add species to this network."""
    self.__graph.add_node(species)