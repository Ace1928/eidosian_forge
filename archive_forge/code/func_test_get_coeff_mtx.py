from chempy.units import units_library, allclose, _sum
from ..testing import requires
@requires('numpy')
def test_get_coeff_mtx():
    r = [({'A': 1}, {'B': 1}), ({'A': 1, 'B': 1}, {'C': 2})]
    A = get_coeff_mtx('ABC', r)
    Aref = np.array([[-1, -1], [1, -1], [0, 2]])
    assert np.allclose(A, Aref)