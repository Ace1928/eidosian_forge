from ..snap.t3mlite import simplex
from ..hyperboloid import *
def print_cell(f):
    if f in simplex.ZeroSubsimplices:
        return 'V'
    if f in simplex.TwoSubsimplices:
        return 'F'
    if f == simplex.T:
        return 'T'
    raise Exception('BLAH')