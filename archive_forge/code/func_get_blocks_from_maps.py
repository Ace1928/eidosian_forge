from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
@deprecated(msg='``get_blocks_from_maps`` is deprecated. This functionality has been incorporated into ``block_triangularize``.', version='6.5.0')
def get_blocks_from_maps(row_block_map, col_block_map):
    blocks = set(row_block_map.values())
    assert blocks == set(col_block_map.values())
    n_blocks = len(blocks)
    block_rows = [[] for _ in range(n_blocks)]
    block_cols = [[] for _ in range(n_blocks)]
    for r, b in row_block_map.items():
        block_rows[b].append(r)
    for c, b in col_block_map.items():
        block_cols[b].append(c)
    return (block_rows, block_cols)