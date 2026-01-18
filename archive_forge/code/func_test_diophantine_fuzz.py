import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
@pytest.mark.slow
def test_diophantine_fuzz():
    rng = np.random.RandomState(1234)
    max_int = np.iinfo(np.intp).max
    for ndim in range(10):
        feasible_count = 0
        infeasible_count = 0
        min_count = 500 // (ndim + 1)
        while min(feasible_count, infeasible_count) < min_count:
            A_max = 1 + rng.randint(0, 11, dtype=np.intp) ** 6
            U_max = rng.randint(0, 11, dtype=np.intp) ** 6
            A_max = min(max_int, A_max)
            U_max = min(max_int - 1, U_max)
            A = tuple((int(rng.randint(1, A_max + 1, dtype=np.intp)) for j in range(ndim)))
            U = tuple((int(rng.randint(0, U_max + 2, dtype=np.intp)) for j in range(ndim)))
            b_ub = min(max_int - 2, sum((a * ub for a, ub in zip(A, U))))
            b = int(rng.randint(-1, b_ub + 2, dtype=np.intp))
            if ndim == 0 and feasible_count < min_count:
                b = 0
            X = solve_diophantine(A, U, b)
            if X is None:
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is None, (A, U, b, X_simplified))
                ranges = tuple((range(0, a * ub + 1, a) for a, ub in zip(A, U)))
                size = 1
                for r in ranges:
                    size *= len(r)
                if size < 100000:
                    assert_(not any((sum(w) == b for w in itertools.product(*ranges))))
                    infeasible_count += 1
            else:
                X_simplified = solve_diophantine(A, U, b, simplify=1)
                assert_(X_simplified is not None, (A, U, b, X_simplified))
                assert_(sum((a * x for a, x in zip(A, X))) == b)
                assert_(all((0 <= x <= ub for x, ub in zip(X, U))))
                feasible_count += 1