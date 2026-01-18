from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def sink_interactions(self, species):
    """Return list of (sink, interaction) pairs for species."""
    return self.__graph.child_edges(species)