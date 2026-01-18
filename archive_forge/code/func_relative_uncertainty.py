import numpy as np
from . import markup
from .quantity import Quantity, scale_other_units
from .registry import unit_registry
from .decorators import with_doc
@property
def relative_uncertainty(self):
    return self.uncertainty.magnitude / self.magnitude