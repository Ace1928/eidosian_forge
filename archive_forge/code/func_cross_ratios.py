from snappy.dev.extended_ptolemy import extended
from snappy.dev.extended_ptolemy import giac_rur
from snappy.dev.extended_ptolemy.complexVolumesClosed import evaluate_at_roots
from snappy.ptolemy.coordinates import PtolemyCoordinates, CrossRatios
def cross_ratios(M):
    return [ptolemys.cross_ratios() for ptolemys in ptolemy_coordinates(M)]