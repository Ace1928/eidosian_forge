from .fixed_points import r13_fixed_points_of_psl2c_matrix # type: ignore
from ..hyperboloid import r13_dot, o13_inverse # type: ignore
from ..upper_halfspace import psl2c_to_o13 # type: ignore
from ..math_basics import is_RealIntervalFieldElement # type: ignore
from ..sage_helper import _within_sage # type: ignore
@staticmethod
def from_psl2c_matrix(m):
    """
        Given a loxodromic PSL(2,C)-matrix m, returns the line (together
        with the O(1,3)-matrix corresponding to m) fixed by m in
        the hyperboloid model.
        """
    return R13LineWithMatrix(R13Line(r13_fixed_points_of_psl2c_matrix(m)), psl2c_to_o13(m))