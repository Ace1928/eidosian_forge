from .simplex import *
from .linalg import Matrix
def boundary_two(manifold):
    VerticesOfFace = {F0: (V1, V2, V3), F1: (V0, V3, V2), F2: (V0, V1, V3), F3: (V0, V2, V1)}
    E, F = (len(manifold.Edges), len(manifold.Faces))
    ans = Matrix(E, F)
    for F in manifold.Faces:
        C = F.Corners[0]
        tet = C.Tetrahedron
        vertices = VerticesOfFace[C.Subsimplex]
        for i in range(3):
            a, b = (vertices[i], vertices[(i + 1) % 3])
            e = tet.Class[a | b]
            ans[e.index(), F.Index] += e.orientation_with_respect_to(tet, a, b)
    return ans