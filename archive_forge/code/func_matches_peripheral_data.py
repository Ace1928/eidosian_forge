import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def matches_peripheral_data(isom):
    Id = matrix(ZZ, [[1, 0], [0, 1]])
    cusp_perm = isom.cusp_images()
    cusp_maps = isom.cusp_maps()
    n = len(cusp_perm)
    return cusp_perm == range(n) and cusp_maps == n * [Id]