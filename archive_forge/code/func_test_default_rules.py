import itertools
from numba.core import types
from numba.core.typeconv.typeconv import TypeManager, TypeCastingRules
from numba.core.typeconv import rules
from numba.core.typeconv import castgraph, Conversion
import unittest
def test_default_rules(self):
    tm = rules.default_type_manager
    self.check_number_compatibility(tm.check_compatible)