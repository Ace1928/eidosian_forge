from .links_base import Strand, Crossing, Link
import random
import collections
def orient_pres_isometric(A, B):
    for iso in A.is_isometric_to(B, True):
        mat = iso.cusp_maps()[0]
        if mat.det() == 1:
            return True
    return False