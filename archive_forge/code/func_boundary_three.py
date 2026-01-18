from .simplex import *
from .linalg import Matrix
def boundary_three(manifold):
    F, T = (len(manifold.Faces), len(manifold.Tetrahedra))
    ans = Matrix(F, T)
    for F in manifold.Faces:
        t0, t1 = [C.Tetrahedron.Index for C in F.Corners]
        ans[F.Index, t0] += 1
        ans[F.Index, t1] += -1
    return ans