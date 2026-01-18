from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def add_interaction(self, source, sink, interaction):
    """Add interaction to this network."""
    self.__graph.add_edge(source, sink, interaction)