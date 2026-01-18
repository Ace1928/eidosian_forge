import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def is_generic(v, non_generic):
    """Checks whether a given variable occurs in a list of non-generic variables

    Note that a variables in such a list may be instantiated to a type term,
    in which case the variables contained in the type term are considered
    non-generic.

    Note: Must be called with v pre-pruned

    Args:
        v: The TypeVariable to be tested for genericity
        non_generic: A set of non-generic TypeVariables

    Returns:
        True if v is a generic variable, otherwise False
    """
    return not occurs_in(v, non_generic)