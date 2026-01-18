from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
def map_coords_to_block_triangular_indices(matrix, matching=None):
    row_blocks, col_blocks = block_triangularize(matrix, matching=matching)
    row_idx_map = {r: idx for idx, rblock in enumerate(row_blocks) for r in rblock}
    col_idx_map = {c: idx for idx, cblock in enumerate(col_blocks) for c in cblock}
    return (row_idx_map, col_idx_map)