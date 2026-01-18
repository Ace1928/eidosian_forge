from datetime import date, datetime, timedelta
import networkx as nx
from networkx.algorithms import isomorphism as iso
def provide_g2_path_3edges(self):
    G2 = nx.DiGraph()
    G2.add_edges_from([(0, 1), (1, 2), (2, 3)])
    return G2