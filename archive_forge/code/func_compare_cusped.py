import snappy
import regina
import snappy.snap.t3mlite as t3m
import snappy.snap.t3mlite.spun as spun
def compare_cusped(snappy_manifold):
    tri_data = snappy_manifold._to_string()
    T = spun.Manifold(tri_data)
    t_hashes = sorted((sorted(S.quad_vector()) for S in T.normal_surfaces()))
    t_slopes = sorted((tuple(S.boundary_slopes()) for S in T.normal_surfaces()))
    R = regina.NSnapPeaTriangulation(tri_data)
    t = R.getNumberOfTetrahedra()
    regina_surfaces = list(vertex_surfaces(R))
    r_hashes = sorted((sorted(sorted((int(S.getQuadCoord(i, j).stringValue()) for i in range(t) for j in range(3)))) for S in regina_surfaces))
    r_slopes = sorted(map(regina_boundary_slope, regina_surfaces))
    assert t_hashes == r_hashes and t_slopes == r_slopes