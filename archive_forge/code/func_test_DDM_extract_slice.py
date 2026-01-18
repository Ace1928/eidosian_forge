from sympy.testing.pytest import raises
from sympy.external.gmpy import HAS_GMPY
from sympy.polys import ZZ, QQ
from sympy.polys.matrices.ddm import DDM
from sympy.polys.matrices.exceptions import (
def test_DDM_extract_slice():
    dm = DDM([[ZZ(1), ZZ(2), ZZ(3)], [ZZ(4), ZZ(5), ZZ(6)], [ZZ(7), ZZ(8), ZZ(9)]], (3, 3), ZZ)
    assert dm.extract_slice(slice(0, 3), slice(0, 3)) == dm
    assert dm.extract_slice(slice(1, 3), slice(-2)) == DDM([[4], [7]], (2, 1), ZZ)
    assert dm.extract_slice(slice(1, 3), slice(-2)) == DDM([[4], [7]], (2, 1), ZZ)
    assert dm.extract_slice(slice(2, 3), slice(-2)) == DDM([[ZZ(7)]], (1, 1), ZZ)
    assert dm.extract_slice(slice(0, 2), slice(-2)) == DDM([[1], [4]], (2, 1), ZZ)
    assert dm.extract_slice(slice(-1), slice(-1)) == DDM([[1, 2], [4, 5]], (2, 2), ZZ)
    assert dm.extract_slice(slice(2), slice(3, 4)) == DDM([[], []], (2, 0), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(2)) == DDM([], (0, 2), ZZ)
    assert dm.extract_slice(slice(3, 4), slice(3, 4)) == DDM([], (0, 0), ZZ)