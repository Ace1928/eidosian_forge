import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def cyclic_product(self, bounds, origin, supremum, centroid=True):
    """Generate initial triangulation using cyclic product"""
    vot = tuple(origin)
    vut = tuple(supremum)
    self.V[vot]
    vo = self.V[vot]
    yield vo.x
    self.V[vut].connect(self.V[vot])
    yield vut
    C0x = [[self.V[vot]]]
    a_vo = copy.copy(list(origin))
    a_vo[0] = vut[0]
    a_vo = self.V[tuple(a_vo)]
    self.V[vot].connect(a_vo)
    yield a_vo.x
    C1x = [[a_vo]]
    ab_C = []
    for i, x in enumerate(bounds[1:]):
        C0x.append([])
        C1x.append([])
        try:
            x[1]
            cC0x = [x[:] for x in C0x[:i + 1]]
            cC1x = [x[:] for x in C1x[:i + 1]]
            for j, (VL, VU) in enumerate(zip(cC0x, cC1x)):
                for k, (vl, vu) in enumerate(zip(VL, VU)):
                    a_vl = list(vl.x)
                    a_vu = list(vu.x)
                    a_vl[i + 1] = vut[i + 1]
                    a_vu[i + 1] = vut[i + 1]
                    a_vl = self.V[tuple(a_vl)]
                    vl.connect(a_vl)
                    yield a_vl.x
                    a_vu = self.V[tuple(a_vu)]
                    vu.connect(a_vu)
                    a_vl.connect(a_vu)
                    vl.connect(a_vu)
                    ab_C.append((vl, a_vu))
                    C0x[i + 1].append(vl)
                    C0x[i + 1].append(vu)
                    C1x[i + 1].append(a_vl)
                    C1x[i + 1].append(a_vu)
                    C0x[j].append(a_vl)
                    C1x[j].append(a_vu)
                    yield a_vu.x
            ab_Cc = copy.copy(ab_C)
            for vp in ab_Cc:
                b_v = list(vp[0].x)
                ab_v = list(vp[1].x)
                b_v[i + 1] = vut[i + 1]
                ab_v[i + 1] = vut[i + 1]
                b_v = self.V[tuple(b_v)]
                ab_v = self.V[tuple(ab_v)]
                vp[0].connect(ab_v)
                b_v.connect(ab_v)
                ab_C.append((vp[0], ab_v))
                ab_C.append((b_v, ab_v))
        except IndexError:
            cC0x = C0x[i]
            cC1x = C1x[i]
            VL, VU = (cC0x, cC1x)
            for k, (vl, vu) in enumerate(zip(VL, VU)):
                a_vu = list(vu.x)
                a_vu[i + 1] = vut[i + 1]
                a_vu = self.V[tuple(a_vu)]
                vu.connect(a_vu)
                vl.connect(a_vu)
                ab_C.append((vl, a_vu))
                C0x[i + 1].append(vu)
                C1x[i + 1].append(a_vu)
                a_vu.connect(self.V[vut])
                yield a_vu.x
                ab_Cc = copy.copy(ab_C)
                for vp in ab_Cc:
                    if vp[1].x[i] == vut[i]:
                        ab_v = list(vp[1].x)
                        ab_v[i + 1] = vut[i + 1]
                        ab_v = self.V[tuple(ab_v)]
                        vp[0].connect(ab_v)
                        ab_C.append((vp[0], ab_v))
    try:
        del C0x
        del cC0x
        del C1x
        del cC1x
        del ab_C
        del ab_Cc
    except UnboundLocalError:
        pass
    if centroid:
        vo = self.V[vot]
        vs = self.V[vut]
        vo.disconnect(vs)
        vc = self.split_edge(vot, vut)
        for v in vo.nn:
            v.connect(vc)
        yield vc.x
        return vc.x
    else:
        yield vut
        return vut