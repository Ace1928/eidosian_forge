from .component import NonZeroDimensionalComponent
from . import processFileBase
from . import processRurFile
from . import utilities
from . import coordinates
from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
def remove_optional_quotes(s):
    if s[0] in ['"', "'"]:
        assert s[0] == s[-1]
        return s[1:-1]
    return s