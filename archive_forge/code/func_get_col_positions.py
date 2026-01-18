import math
import sys
from Bio import MissingPythonDependencyError
def get_col_positions(tree):
    """Create a mapping of each clade to its column position."""
    depths = tree.depths()
    if max(depths.values()) == 0:
        depths = tree.depths(unit_branch_lengths=True)
    fudge_margin = int(math.ceil(math.log(len(taxa), 2)))
    cols_per_branch_unit = (drawing_width - fudge_margin) / max(depths.values())
    return {clade: int(blen * cols_per_branch_unit + 1.0) for clade, blen in depths.items()}