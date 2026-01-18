from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def barycentric_face_embedding(arrow, north_pole=None):
    """
    The arrow here is a directed edge in a specified tetrahedron.  It
    also specifies a face (the face which is disjoint from the arrow,
    except for the head).  This helper function takes the arrow and
    embeds the face of the arrow in the xy-plane, with the two
    tetrahedra on either side of the face embedded in the upper and
    lower half-spaces.  The specific coordinates are labeled below; A,
    B, and C are the vertices of the image of the face in the
    xy-plane, and N and S are images of the vertices of the two
    tetrahedron not in the face.
    """
    if north_pole is None:
        north_pole = N
    next_arrow = arrow.glued()
    top_bdry = [None, 't1', 't2', 't3']
    bottom_brdy = ['b1', None, 'b2', 'b3']
    emb_top = tetrahedron_embedding(arrow, [north_pole, A, C, B], top_bdry)
    emb_bottom = tetrahedron_embedding(next_arrow, [A, S, C, B], bottom_brdy)
    return [(arrow.Tetrahedron, emb_top), (next_arrow.Tetrahedron, emb_bottom)]