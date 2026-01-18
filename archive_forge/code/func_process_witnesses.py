from .component import NonZeroDimensionalComponent
from . import processFileBase
from . import processRurFile
from . import utilities
from . import coordinates
from .polynomial import Polynomial
from .ptolemyVarietyPrimeIdealGroebnerBasis import PtolemyVarietyPrimeIdealGroebnerBasis
def process_witnesses(py_eval, manifold_thunk, witnesses_section, for_dimension, variables):
    return [process_solutions_provider(py_eval, manifold_thunk, witness_section, for_dimension, variables) for witness_section in processFileBase.find_section(witnesses_section, 'WITNESS')]