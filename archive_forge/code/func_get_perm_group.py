from collections import defaultdict
from enum import Enum
import itertools
from sympy.combinatorics.named_groups import (
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.combinatorics.permutations import Permutation
def get_perm_group(self):
    if self == S6TransitiveSubgroups.C6:
        return CyclicGroup(6)
    elif self == S6TransitiveSubgroups.S3:
        return S3_in_S6()
    elif self == S6TransitiveSubgroups.D6:
        return DihedralGroup(6)
    elif self == S6TransitiveSubgroups.A4:
        return A4_in_S6()
    elif self == S6TransitiveSubgroups.G18:
        return G18()
    elif self == S6TransitiveSubgroups.A4xC2:
        return A4xC2()
    elif self == S6TransitiveSubgroups.S4m:
        return S4m()
    elif self == S6TransitiveSubgroups.S4p:
        return S4p()
    elif self == S6TransitiveSubgroups.G36m:
        return G36m()
    elif self == S6TransitiveSubgroups.G36p:
        return G36p()
    elif self == S6TransitiveSubgroups.S4xC2:
        return S4xC2()
    elif self == S6TransitiveSubgroups.PSL2F5:
        return PSL2F5()
    elif self == S6TransitiveSubgroups.G72:
        return G72()
    elif self == S6TransitiveSubgroups.PGL2F5:
        return PGL2F5()
    elif self == S6TransitiveSubgroups.A6:
        return AlternatingGroup(6)
    elif self == S6TransitiveSubgroups.S6:
        return SymmetricGroup(6)