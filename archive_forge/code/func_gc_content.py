import math
import numbers
import numpy as np
from Bio.Seq import Seq
from . import _pwm  # type: ignore
@property
def gc_content(self):
    """Compute the GC-ratio."""
    raise Exception('Cannot compute the %GC composition of a PSSM')