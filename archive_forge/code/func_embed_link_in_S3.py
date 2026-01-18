import itertools
from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import QQ, Vector3, Vector4, Matrix
from .barycentric_geometry import (BarycentricPoint,
from .mcomplex_with_link import McomplexWithLink
def embed_link_in_S3(M):
    """
    >>> K = example10()
    >>> components = embed_link_in_S3(K)
    >>> len(components), len(components[0])
    (1, 19)
    """
    target = 'cMcabbgdv'
    assert M.isosig() == target
    K = transfer_arcs(M, McomplexWithLink(target))
    ans = []
    for arcs in K.link_components():
        component = []
        for arc in arcs:
            v = arc.start.vector
            if isinstance(arc, InfinitesimalArc):
                if arc.start_tet.Index == 0:
                    component.append(tet0_map * v)
                    if v[0] == 0:
                        face_cor = Vector3([v[2], v[3], v[1]])
                        component.append(fin_top_map * face_cor)
                        assert arc.end_tet.Index == 0
                    elif v[1] == 0:
                        face_cor = Vector3([v[2], v[3], v[0]])
                        component.append(fin_top_map * face_cor)
                        assert arc.end_tet.Index == 0
                    elif v[2] == 0:
                        face_cor = Vector3([v[0], v[1], v[3]])
                        component.append(fin_right_map * face_cor)
                    elif v[3] == 0:
                        face_cor = Vector3([v[0], v[2], v[1]])
                        component.append(fin_left_map * face_cor)
                else:
                    component.append(tet1_map * v)
                    if v[0] == 0:
                        face_cor = Vector3([v[1], v[3], v[2]])
                        component.append(fin_bottom_map * face_cor)
                    elif v[1] == 0:
                        face_cor = Vector3([v[0], v[2], v[3]])
                        component.append(fin_left_map * face_cor)
                    elif v[2] == 0:
                        face_cor = Vector3([v[1], v[3], v[0]])
                        component.append(fin_bottom_map * face_cor)
                    elif v[3] == 0:
                        face_cor = Vector3([v[0], v[1], v[2]])
                        component.append(fin_right_map * face_cor)
            else:
                tet_map = tet0_map if arc.tet.Index == 0 else tet1_map
                component.append(tet_map * v)
        ans.append(component)
    return ans