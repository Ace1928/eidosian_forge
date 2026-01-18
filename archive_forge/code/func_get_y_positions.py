import math
import sys
from Bio import MissingPythonDependencyError
def get_y_positions(tree):
    """Create a mapping of each clade to its vertical position.

        Dict of {clade: y-coord}.
        Coordinates are negative, and integers for tips.
        """
    maxheight = tree.count_terminals()
    heights = {tip: maxheight - i for i, tip in enumerate(reversed(tree.get_terminals()))}

    def calc_row(clade):
        for subclade in clade:
            if subclade not in heights:
                calc_row(subclade)
        heights[clade] = (heights[clade.clades[0]] + heights[clade.clades[-1]]) / 2.0
    if tree.root.clades:
        calc_row(tree.root)
    return heights