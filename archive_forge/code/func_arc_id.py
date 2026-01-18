import pickle
from .links import Crossing, Strand, Link
from . import planar_isotopy
def arc_id(c, i):
    """Get the unique integer id associated to the arc, generating
            a fresh one if needed."""
    return arc_ids.setdefault(arc_key(c, i), len(arc_ids) + 1)