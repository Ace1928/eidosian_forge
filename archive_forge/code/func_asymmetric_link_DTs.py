import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def asymmetric_link_DTs(N):
    found = 0
    while found < N:
        M = snappy.HTLinkExteriors.random()
        if appears_hyperbolic(M) and M.symmetry_group().order() == 1:
            found += 1
            yield M.DT_code()