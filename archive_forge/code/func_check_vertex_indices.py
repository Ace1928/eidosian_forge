from ..snap.t3mlite import simplex
from ..hyperboloid import *
def check_vertex_indices(tets):
    for tet in tets:
        for v in simplex.ZeroSubsimplices:
            index = tet.post_drill_infos[v]
            for f in simplex.TwoSubsimplices:
                if v & f:
                    if tet.Neighbor[f].post_drill_infos[tet.Gluing[f].image(v)] != index:
                        print('tet, v face:', tet, v, f)
                        print('index and other index:', index, tet.Neighbor[f].post_drill_infos, [tet.Gluing[f].image(v)])
                        raise Exception("Neighbors don't have same vertex.")