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
def test_flattenings_from_tetrahedra_shapes_of_manifold():
    old_precision = pari.set_real_precision(100)
    p = pari('Pi * Pi / 6')

    def is_close(cvol1, cvol2, epsilon):
        diff = cvol1 - cvol2
        return diff.real().abs() < epsilon and (diff.imag() % p < epsilon or -diff.imag() % p < epsilon)
    from snappy import OrientableCuspedCensus
    for M in list(OrientableCuspedCensus()[0:10]) + list(OrientableCuspedCensus()[10000:10010]):
        flattening = Flattenings.from_tetrahedra_shapes_of_manifold(M)
        flattening.check_against_manifold(M, epsilon=1e-80)
        if not is_close(flattening.complex_volume(), M.complex_volume(), epsilon=1e-13):
            raise Exception('Wrong volume')
    M = ManifoldGetter('5_2')
    flattening = Flattenings.from_tetrahedra_shapes_of_manifold(M)
    flattening.check_against_manifold(M, epsilon=1e-80)
    if not is_close(flattening.complex_volume(), pari('2.828122088330783162763898809276634942770981317300649477043520327258802548322471630936947017929999108 - 3.024128376509301659719951221694600993984450242270735312503300643508917708286193746506469158300472966*I'), epsilon=1e-80):
        raise Exception('Wrong volume')
    pari.set_real_precision(old_precision)