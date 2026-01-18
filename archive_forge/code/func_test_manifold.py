from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def test_manifold(manifold):
    G = snap.polished_holonomy(manifold, dec_prec=dec_prec)