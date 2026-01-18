from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import Matrix, Vector3, Vector4
from . import pl_utils
def barycentric_quad_embedding1(arrow, north_pole=None):
    """
    Take an arrow with 4 valent axis, then build embedding of the
    surrounding four tetrahedra forming an octahedron in R^3 using the
    arrows running around the valence 4 edge in zy plane.
    """
    n = Vector3([0, 0, 1]) if north_pole is None else north_pole
    e = Vector3([1, 0, 0])
    s = Vector3([0, 0, -1])
    w = Vector3([-1, 0, 0])
    a = Vector3([0, -1, 0])
    b = Vector3([0, 1, 0])
    arrow = arrow.copy()
    ans = []
    verts = [e, s, w, n, e]
    for i in range(4):
        ans.append((arrow.Tetrahedron, tetrahedron_embedding(arrow, [verts[i], verts[i + 1], a, b])))
        arrow.next()
    return ans