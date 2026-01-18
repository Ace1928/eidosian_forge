import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def plink_component_order_test(N):
    M2 = snappy.Manifold()
    return set((test_DT(dt, M2) for dt in asymmetric_link_DTs(N)))