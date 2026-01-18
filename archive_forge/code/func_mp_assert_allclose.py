import os
import sys
import time
from itertools import zip_longest
import numpy as np
from numpy.testing import assert_
import pytest
from scipy.special._testutils import assert_func_equal
def mp_assert_allclose(res, std, atol=0, rtol=1e-17):
    """
    Compare lists of mpmath.mpf's or mpmath.mpc's directly so that it
    can be done to higher precision than double.
    """
    failures = []
    for k, (resval, stdval) in enumerate(zip_longest(res, std)):
        if resval is None or stdval is None:
            raise ValueError('Lengths of inputs res and std are not equal.')
        if mpmath.fabs(resval - stdval) > atol + rtol * mpmath.fabs(stdval):
            failures.append((k, resval, stdval))
    nfail = len(failures)
    if nfail > 0:
        ndigits = int(abs(np.log10(rtol)))
        msg = ['']
        msg.append(f'Bad results ({nfail} out of {k + 1}) for the following points:')
        for k, resval, stdval in failures:
            resrep = mpmath.nstr(resval, ndigits, min_fixed=0, max_fixed=0)
            stdrep = mpmath.nstr(stdval, ndigits, min_fixed=0, max_fixed=0)
            if stdval == 0:
                rdiff = 'inf'
            else:
                rdiff = mpmath.fabs((resval - stdval) / stdval)
                rdiff = mpmath.nstr(rdiff, 3)
            msg.append(f'{k}: {resrep} != {stdrep} (rdiff {rdiff})')
        assert_(False, '\n'.join(msg))