from collections import defaultdict
import networkx as nx
def conflicting(self, b, planarity_state):
    """Returns True if interval I conflicts with edge b"""
    return not self.empty() and planarity_state.lowpt[self.high] > planarity_state.lowpt[b]