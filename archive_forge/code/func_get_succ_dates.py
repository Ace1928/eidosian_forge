import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def get_succ_dates(self, Gx, Gx_node, core_x, succ):
    """
        Get the dates of edges to successors.
        """
    succ_dates = []
    if isinstance(Gx, nx.DiGraph):
        for n in succ:
            succ_dates.append(Gx[Gx_node][n][self.temporal_attribute_name])
    else:
        for n in succ:
            for edge in Gx[Gx_node][n].values():
                succ_dates.append(edge[self.temporal_attribute_name])
    return succ_dates