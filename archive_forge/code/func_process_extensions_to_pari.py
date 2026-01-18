from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def process_extensions_to_pari(extensions):
    """
    Similar to _process_extensions but returns pari objects.
    """
    number_field, ext_assignments = _process_extensions(extensions)
    return _number_field_and_ext_assignment_to_pari(number_field, ext_assignments)