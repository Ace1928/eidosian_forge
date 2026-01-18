from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
@deprecated('\n    Importing default_sort_key from sympy.utilities.iterables is deprecated.\n    Use from sympy import default_sort_key instead.\n    ', deprecated_since_version='1.10', active_deprecations_target='deprecated-sympy-core-compatibility')
def default_sort_key(*args, **kwargs):
    from sympy import default_sort_key as _default_sort_key
    return _default_sort_key(*args, **kwargs)