from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
@deprecated(msg='``get_diagonal_blocks`` has been deprecated. Please use ``block_triangularize`` instead.', version='6.5.0')
def get_diagonal_blocks(matrix, matching=None):
    return block_triangularize(matrix, matching=matching)