import weakref
import numpy
from .dimensionality import Dimensionality
from . import markup
from .quantity import Quantity, get_conversion_factor
from .registry import unit_registry
from .decorators import memoize, with_doc
@classmethod
def get_default_unit(cls):
    return cls._default_unit