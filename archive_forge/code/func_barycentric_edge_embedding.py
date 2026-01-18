from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def barycentric_edge_embedding(arrow, north_pole=None):
    """
    Take the arrow corresponding to an edge of valence 3. This
    function then creates an embedding of the three tetrahedra glued
    in pairs around the edge into R^3. The embedding is defined so
    that the edge goes from N to S, as labeled below, The arrow goes
    from A to B; A, B, and C form a triangle in the xy-plane.

    Note that this arrangement has the same image in R^3 as the
    barycentric_face_embedding above -- that's by design, so that we
    can use these two maps to transfer the arcs in barycentric
    coordinates under two-three and two-three moves.
    """
    if north_pole is None:
        north_pole = N
    assert len(arrow.linking_cycle()) == 3
    arrow = arrow.copy()
    verts = [A, B, C, A]
    ans = []
    for i in range(3):
        tet_verts = [verts[i], verts[i + 1], S, north_pole]
        bdry_map = [None, None, f't{i + 1}', f'b{i + 1}']
        ans.append((arrow.Tetrahedron, tetrahedron_embedding(arrow, tet_verts, bdry_map)))
        arrow.next()
    return ans