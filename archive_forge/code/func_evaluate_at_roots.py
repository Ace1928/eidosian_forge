from snappy.verify.complex_volume.adjust_torsion import (
from snappy.verify.complex_volume.closed import zero_lifted_holonomy
from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
import snappy.snap.t3mlite as t3m
from sage.all import (RealIntervalField, ComplexIntervalField,
import sage.all
import re
def evaluate_at_roots(numberField, exact_values, precision=53):
    """
    numberField is a sage number field.
    exact_values a dictionary where values are elements in that number field.
    precision is desired precision in bits.

    For each embedding of the number field, evaluates the values in the
    dictionary and produces a new dictionary with the same keys.
    The new dictionaries are returned in a list.
    """
    CIF = ComplexIntervalField(precision)
    return [{k: v.lift().substitute(root) for k, v in exact_values.items()} for root, multiplicity in numberField.polynomial().roots(CIF)]