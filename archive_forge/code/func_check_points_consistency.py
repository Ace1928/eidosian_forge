from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_points_consistency(m):
    for tet in m.Tetrahedra:
        for F in simplex.TwoSubsimplices:
            for V in simplex.ZeroSubsimplices:
                if V & F:
                    check_points_equal(tet.O13_matrices[F] * tet.R13_vertices[V], tet.Neighbor[F].R13_vertices[tet.Gluing[F].image(V)])