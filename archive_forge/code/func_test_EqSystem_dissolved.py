import collections
import pytest
from ..util.testing import requires
from ..chemistry import Substance, Reaction, Equilibrium, Species
@requires('numpy')
def test_EqSystem_dissolved():
    eqsys, names, _ = _get_NaCl(Cls=Species, phase_idx=1)
    inp = eqsys.as_per_substance_array({'Na+': 1, 'Cl-': 2, 'NaCl': 4})
    result = eqsys.dissolved(inp)
    ref = eqsys.as_per_substance_array({'Na+': 5, 'Cl-': 6, 'NaCl': 0})
    assert np.allclose(result, ref)