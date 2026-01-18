from functools import partial
from .constants import DefaultValue
def check_trait(trait):
    """ Returns either the original value or a valid CTrait if the value can be
        converted to a CTrait.
    """
    try:
        return as_ctrait(trait)
    except TypeError:
        return trait