from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_consistency_segments(segments):
    for i in range(len(segments)):
        s0 = segments[i]
        s1 = segments[(i + 1) % len(segments)]
        if s0.tet.Class[s0.endpoints[1].subsimplex] is not s1.tet.Class[s1.endpoints[0].subsimplex]:
            raise Exception('Classes of consecutive segments not matching %i' % i)
        if s0.next_ is not s1:
            raise Exception('Linked list broken (next)')
        if s1.prev is not s0:
            raise Exception('Linked list broken (prev)')
        if s0.endpoints[1].subsimplex in simplex.TwoSubsimplices:
            check_points_equal(s0.tet.O13_matrices[s0.endpoints[1].subsimplex] * s0.endpoints[1].r13_point, s1.endpoints[0].r13_point)