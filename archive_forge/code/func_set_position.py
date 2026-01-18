from collections import defaultdict
import networkx as nx
def set_position(parent, tree, remaining_nodes, delta_x, y_coordinate, pos):
    """Helper method to calculate the absolute position of nodes."""
    child = tree[parent]
    parent_node_x = pos[parent][0]
    if child is not None:
        child_x = parent_node_x + delta_x[child]
        pos[child] = (child_x, y_coordinate[child])
        remaining_nodes.append(child)