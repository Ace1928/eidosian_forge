from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def glue_to(self, next_arc):
    """
        Helper method used when concatenating two linked lists of Arcs.

        Assumes self.end == next_arc.start and then makes them the
        same object.
        """
    self.next = next_arc
    next_arc.past = self
    next_arc.start = self.end