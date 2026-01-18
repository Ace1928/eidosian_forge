from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import vector, RealDoubleField, sqrt
def normalize_gram_matrices(ms):
    return [normalize_gram_matrix(m) for m in ms]