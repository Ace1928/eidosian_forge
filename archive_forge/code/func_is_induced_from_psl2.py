from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def is_induced_from_psl2(self, epsilon=None):
    """
        For each simplex and each edges, checks that all cross ratios of that
        simplex that are parallel to that each are the same (maximal absolute
        difference is the epsilon given as argument).
        This means that the corresponding representation is induced by a
        PSL(2,C) representation.
        """
    d = {}
    for key, value in self.items():
        variable_name, index, tet_index = key.split('_')
        if variable_name not in ['z', 'zp', 'zpp']:
            raise Exception('Variable not z, zp, or, zpp')
        if len(index) != 4:
            raise Exception('Not 4 indices')
        short_key = variable_name + '_' + tet_index
        old_value = d.setdefault(short_key, value)
        if epsilon is None:
            if value != old_value:
                return False
        elif (value - old_value).abs() > epsilon:
            return False
    return True