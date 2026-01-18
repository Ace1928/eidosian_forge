from collections import defaultdict
import networkx as nx
def lowest(self, planarity_state):
    """Returns the lowest lowpoint of a conflict pair"""
    if self.left.empty():
        return planarity_state.lowpt[self.right.low]
    if self.right.empty():
        return planarity_state.lowpt[self.left.low]
    return min(planarity_state.lowpt[self.left.low], planarity_state.lowpt[self.right.low])