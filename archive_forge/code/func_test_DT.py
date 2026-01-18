import snappy
import spherogram
import spherogram.links.orthogonal
from nsnappytools import appears_hyperbolic
from sage.all import *
def test_DT(dt, M2=None):
    if M2 is None:
        M2 = snappy.Manifold()
    dtc = spherogram.DTcodec(dt)
    L = dtc.link()
    M0, M1 = (dtc.exterior(), L.exterior())
    L.view(M2.LE)
    M2.LE.callback()
    return manifolds_match(M0, M1) and manifolds_match(M1, M2)