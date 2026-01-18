import itertools
import unittest
from numba import njit
from numba.core import types
@classmethod
def automatic_populate(cls):
    tys = types.integer_domain | types.real_domain
    for fromty, toty in itertools.permutations(tys, r=2):
        test_name = 'test_{fromty}_to_{toty}'.format(fromty=fromty, toty=toty)
        setattr(cls, test_name, template(fromty, toty))