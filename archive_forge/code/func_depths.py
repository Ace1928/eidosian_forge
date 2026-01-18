import collections
import copy
import itertools
import random
import re
import warnings
def depths(self, unit_branch_lengths=False):
    """Create a mapping of tree clades to depths (by branch length).

        :Parameters:
            unit_branch_lengths : bool
                If True, count only the number of branches (levels in the tree).
                By default the distance is the cumulative branch length leading
                to the clade.

        :returns: dict of {clade: depth}, where keys are all of the Clade
            instances in the tree, and values are the distance from the root to
            each clade (including terminals).

        """
    if unit_branch_lengths:
        depth_of = lambda c: 1
    else:
        depth_of = lambda c: c.branch_length or 0
    depths = {}

    def update_depths(node, curr_depth):
        depths[node] = curr_depth
        for child in node.clades:
            new_depth = curr_depth + depth_of(child)
            update_depths(child, new_depth)
    update_depths(self.root, self.root.branch_length or 0)
    return depths