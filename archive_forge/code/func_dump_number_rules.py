import itertools
from .typeconv import TypeManager, TypeCastingRules
from numba.core import types
def dump_number_rules():
    tm = default_type_manager
    for a, b in itertools.product(types.number_domain, types.number_domain):
        print(a, '->', b, tm.check_compatible(a, b))