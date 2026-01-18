from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def check_against_manifold(self, M=None, epsilon=None):
    """
        Checks that the given solution really is a solution to the PGL(N,C) gluing
        equations of a manifold. Usage similar to check_against_manifold of
        PtolemyCoordinates. See help(ptolemy.PtolemtyCoordinates) for example.

        === Arguments ===

        M --- manifold to check this for
        epsilon --- maximal allowed error when checking the relations, use
        None for exact comparison.
        """
    if M is None:
        M = self.get_manifold()
    if M is None:
        raise Exception('Need to give manifold')
    some_z = list(self.keys())[0]
    variable_name, index, tet_index = some_z.split('_')
    if variable_name not in ['z', 'zp', 'zpp']:
        raise Exception('Variable not z, zp, or, zpp')
    if len(index) != 4:
        raise Exception('Not 4 indices')
    N = sum([int(x) for x in index]) + 2
    matrix_with_explanations = M.gluing_equations_pgl(N, equation_type='all')
    matrix = matrix_with_explanations.matrix
    rows = matrix_with_explanations.explain_rows
    cols = matrix_with_explanations.explain_columns
    for row in range(len(rows)):
        product = 1
        for col in range(len(cols)):
            cross_ratio_variable = cols[col]
            cross_ratio_value = self[cross_ratio_variable]
            product = product * cross_ratio_value ** matrix[row, col]
        _check_relation(product - 1, epsilon, 'Gluing equation %s' % rows[row])