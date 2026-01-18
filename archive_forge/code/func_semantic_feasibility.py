import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def semantic_feasibility(self, G1_node, G2_node):
    """Returns True if adding (G1_node, G2_node) is semantically
        feasible.

        Any subclass which redefines semantic_feasibility() must
        maintain the self.tests if needed, to keep the match() method
        functional. Implementations should consider multigraphs.
        """
    pred, succ = ([n for n in self.G1.predecessors(G1_node) if n in self.core_1], [n for n in self.G1.successors(G1_node) if n in self.core_1])
    if not self.one_hop(self.G1, G1_node, self.core_1, pred, succ):
        return False
    if not self.two_hop_pred(self.G1, G1_node, self.core_1, pred):
        return False
    if not self.two_hop_succ(self.G1, G1_node, self.core_1, succ):
        return False
    return True