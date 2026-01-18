import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def fill_vols(M):
    vols = []
    n = M.num_cusps()
    for i in range(n):
        N = M.without_hyperbolic_structure()
        fill = n * [(1, 0)]
        fill[i] = (0, 0)
        N.dehn_fill(fill)
        N = N.filled_triangulation().with_hyperbolic_structure()
        print(N.num_cusps())
        vols.append(N.volume())
    return vols