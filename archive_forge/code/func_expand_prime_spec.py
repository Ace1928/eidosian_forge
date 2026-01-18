from ..sage_helper import _within_sage, sage_method
from .. import SnapPy
def expand_prime_spec(spec):
    if spec in ZZ:
        a, b = (0, spec)
    else:
        if len(spec) != 2:
            raise ValueError(f'Spec {spec} does not specify a range')
        a, b = spec
    return prime_range(a, b + 1)