from collections import defaultdict
import networkx as nx
def neighbors_cw_order(self, v):
    """Generator for the neighbors of v in clockwise order.

        Parameters
        ----------
        v : node

        Yields
        ------
        node

        """
    if len(self[v]) == 0:
        return
    start_node = self.nodes[v]['first_nbr']
    yield start_node
    current_node = self[v][start_node]['cw']
    while start_node != current_node:
        yield current_node
        current_node = self[v][current_node]['cw']