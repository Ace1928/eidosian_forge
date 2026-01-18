from functools import reduce
from Bio.Pathway.Rep.MultiGraph import MultiGraph
def interactions(self):
    """Return list of the unique interactions in this network."""
    return self.__graph.labels()