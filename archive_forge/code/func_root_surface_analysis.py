from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def root_surface_analysis(primitive_slab, root, eps=1e-08):
    """A tool to analyze a slab and look for valid roots that exist, up to
    the given root. This is useful for generating all possible cells
    without prior knowledge.

    *primitive slab* is the primitive cell to analyze.

    *root* is the desired root to find, and all below."""
    return _root_surface_analysis(primitive_slab=primitive_slab, root=root, eps=eps)[2]