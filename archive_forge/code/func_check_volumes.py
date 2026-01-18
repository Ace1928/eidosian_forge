from snappy import Manifold, pari, ptolemy
from snappy.ptolemy import solutions_from_magma, Flattenings, parse_solutions
from snappy.ptolemy.processFileBase import get_manifold
from snappy.ptolemy import __path__ as ptolemy_paths
from snappy.ptolemy.coordinates import PtolemyCannotBeCheckedError
from snappy.sage_helper import _within_sage, doctest_modules
from snappy.pari import pari
import bz2
import os
import sys
def check_volumes(complex_volumes, baseline_complex_volumes, check_real_part_only=False, torsion_imaginary_part=6, epsilon=1e-80):
    conjugates = [-cvol.conj() for cvol in baseline_complex_volumes]
    baseline_complex_volumes = baseline_complex_volumes + conjugates
    p = pari('Pi * Pi') / torsion_imaginary_part

    def is_close(cvol1, cvol2):
        diff = cvol1 - cvol2
        if diff.real().abs() > epsilon:
            return False
        if check_real_part_only:
            return True
        return diff.imag() % p < epsilon or -diff.imag() % p < epsilon
    for cvol1 in baseline_complex_volumes:
        if not any((is_close(cvol1, cvol2) for cvol2 in complex_volumes)):
            print('Missing base line volume:', cvol1)
            print('Volumes:')
            for i in complex_volumes:
                print('     ', i)
            raise Exception
    for cvol2 in complex_volumes:
        if not any((is_close(cvol1, cvol2) for cvol1 in baseline_complex_volumes)):
            print('Extra volume:', cvol2)
            print('Volumes:')
            for i in complex_volumes:
                print('     ', i)
            raise Exception