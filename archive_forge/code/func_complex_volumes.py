from snappy.verify.complex_volume.adjust_torsion import (
from snappy.verify.complex_volume.closed import zero_lifted_holonomy
from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
import snappy.snap.t3mlite as t3m
from sage.all import (RealIntervalField, ComplexIntervalField,
import sage.all
import re
def complex_volumes(M, precision=53):
    """
    Compute all complex volumes from the extended Ptolemy variety for the
    closed manifold M (given as Dehn-filling on 1-cusped manifold).
    Note: not every volume might correspond to a representation factoring
    through the closed manifold. In particular, we get the complex volume
    of the geometric representation of the cusped manifold.
    """
    representative_ptolemys, full_var_dict = compute_representative_ptolemys_and_full_var_dict(M, precision)
    return [[verified_complex_volume_from_lifted_ptolemys(t3m.Mcomplex(M), lift_ptolemy_coordinates(M, sol, full_var_dict)) for sol in galois_conjugates] for galois_conjugates in representative_ptolemys]