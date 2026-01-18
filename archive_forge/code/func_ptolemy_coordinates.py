from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
from snappy.dev.extended_ptolemy.complexVolumesClosed import evaluate_at_roots
from snappy.ptolemy.coordinates import PtolemyCoordinates, CrossRatios
def ptolemy_coordinates(M):
    return [PtolemyCoordinates(d, is_numerical=False, manifold_thunk=lambda: M) for nf, d, multiplicity in extended_ptolemy_solutions(M)]