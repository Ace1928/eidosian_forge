import logging
import math
from nltk.parse.dependencygraph import DependencyGraph
def compute_original_indexes(self, new_indexes):
    """
        As nodes are collapsed into others, they are replaced
        by the new node in the graph, but it's still necessary
        to keep track of what these original nodes were.  This
        takes a list of node addresses and replaces any collapsed
        node addresses with their original addresses.

        :type new_indexes: A list of integers.
        :param new_indexes: A list of node addresses to check for
            subsumed nodes.
        """
    swapped = True
    while swapped:
        originals = []
        swapped = False
        for new_index in new_indexes:
            if new_index in self.inner_nodes:
                for old_val in self.inner_nodes[new_index]:
                    if old_val not in originals:
                        originals.append(old_val)
                        swapped = True
            else:
                originals.append(new_index)
        new_indexes = originals
    return new_indexes