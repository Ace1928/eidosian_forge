import copy
import logging
import itertools
import decimal
from functools import cache
import numpy
from ._vertex import (VertexCacheField, VertexCacheIndex)
def refine_local_space(self, origin, supremum, bounds, centroid=1):
    origin_c = copy.copy(origin)
    supremum_c = copy.copy(supremum)
    vl, vu, a_vu = (None, None, None)
    s_ov = list(origin)
    s_origin = list(origin)
    s_sv = list(supremum)
    s_supremum = list(supremum)
    for i, vi in enumerate(s_origin):
        if s_ov[i] > s_sv[i]:
            s_origin[i] = s_sv[i]
            s_supremum[i] = s_ov[i]
    vot = tuple(s_origin)
    vut = tuple(s_supremum)
    vo = self.V[vot]
    vs = self.V[vut]
    vco = self.split_edge(vo.x, vs.x)
    sup_set = copy.copy(vco.nn)
    a_vl = copy.copy(list(vot))
    a_vl[0] = vut[0]
    if tuple(a_vl) not in self.V.cache:
        vo = self.V[vot]
        vs = self.V[vut]
        vco = self.split_edge(vo.x, vs.x)
        sup_set = copy.copy(vco.nn)
        a_vl = copy.copy(list(vot))
        a_vl[0] = vut[0]
        a_vl = self.V[tuple(a_vl)]
    else:
        a_vl = self.V[tuple(a_vl)]
    c_v = self.split_edge(vo.x, a_vl.x)
    c_v.connect(vco)
    yield c_v.x
    Cox = [[vo]]
    Ccx = [[c_v]]
    Cux = [[a_vl]]
    ab_C = []
    s_ab_C = []
    for i, x in enumerate(bounds[1:]):
        Cox.append([])
        Ccx.append([])
        Cux.append([])
        try:
            t_a_vl = list(vot)
            t_a_vl[i + 1] = vut[i + 1]
            cCox = [x[:] for x in Cox[:i + 1]]
            cCcx = [x[:] for x in Ccx[:i + 1]]
            cCux = [x[:] for x in Cux[:i + 1]]
            ab_Cc = copy.copy(ab_C)
            s_ab_Cc = copy.copy(s_ab_C)
            if tuple(t_a_vl) not in self.V.cache:
                raise IndexError
            t_a_vu = list(vut)
            t_a_vu[i + 1] = vut[i + 1]
            if tuple(t_a_vu) not in self.V.cache:
                raise IndexError
            for vectors in s_ab_Cc:
                bc_vc = list(vectors[0].x)
                b_vl = list(vectors[1].x)
                b_vu = list(vectors[2].x)
                ba_vu = list(vectors[3].x)
                bc_vc[i + 1] = vut[i + 1]
                b_vl[i + 1] = vut[i + 1]
                b_vu[i + 1] = vut[i + 1]
                ba_vu[i + 1] = vut[i + 1]
                bc_vc = self.V[tuple(bc_vc)]
                bc_vc.connect(vco)
                yield bc_vc
                d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                d_bc_vc.connect(bc_vc)
                d_bc_vc.connect(vectors[1])
                d_bc_vc.connect(vectors[2])
                d_bc_vc.connect(vectors[3])
                yield d_bc_vc.x
                b_vl = self.V[tuple(b_vl)]
                bc_vc.connect(b_vl)
                d_bc_vc.connect(b_vl)
                yield b_vl
                b_vu = self.V[tuple(b_vu)]
                bc_vc.connect(b_vu)
                d_bc_vc.connect(b_vu)
                b_vl_c = self.split_edge(b_vu.x, b_vl.x)
                bc_vc.connect(b_vl_c)
                yield b_vu
                ba_vu = self.V[tuple(ba_vu)]
                bc_vc.connect(ba_vu)
                d_bc_vc.connect(ba_vu)
                os_v = self.split_edge(vectors[1].x, ba_vu.x)
                ss_v = self.split_edge(b_vl.x, ba_vu.x)
                b_vu_c = self.split_edge(b_vu.x, ba_vu.x)
                bc_vc.connect(b_vu_c)
                yield os_v.x
                yield ss_v.x
                yield ba_vu
                d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                d_bc_vc.connect(vco)
                yield d_bc_vc.x
                d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_b_vl)
                yield d_b_vl.x
                d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_b_vu)
                yield d_b_vu.x
                d_ba_vu = self.split_edge(vectors[3].x, ba_vu.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_ba_vu)
                yield d_ba_vu
                comb = [vl, vu, a_vu, b_vl, b_vu, ba_vu]
                comb_iter = itertools.combinations(comb, 2)
                for vecs in comb_iter:
                    self.split_edge(vecs[0].x, vecs[1].x)
                ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                ab_C.append((d_bc_vc, vl, b_vl, a_vu, ba_vu))
            for vectors in ab_Cc:
                bc_vc = list(vectors[0].x)
                b_vl = list(vectors[1].x)
                b_vu = list(vectors[2].x)
                ba_vl = list(vectors[3].x)
                ba_vu = list(vectors[4].x)
                bc_vc[i + 1] = vut[i + 1]
                b_vl[i + 1] = vut[i + 1]
                b_vu[i + 1] = vut[i + 1]
                ba_vl[i + 1] = vut[i + 1]
                ba_vu[i + 1] = vut[i + 1]
                bc_vc = self.V[tuple(bc_vc)]
                bc_vc.connect(vco)
                yield bc_vc
                d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                d_bc_vc.connect(bc_vc)
                d_bc_vc.connect(vectors[1])
                d_bc_vc.connect(vectors[2])
                d_bc_vc.connect(vectors[3])
                d_bc_vc.connect(vectors[4])
                yield d_bc_vc.x
                b_vl = self.V[tuple(b_vl)]
                bc_vc.connect(b_vl)
                d_bc_vc.connect(b_vl)
                yield b_vl
                b_vu = self.V[tuple(b_vu)]
                bc_vc.connect(b_vu)
                d_bc_vc.connect(b_vu)
                yield b_vu
                ba_vl = self.V[tuple(ba_vl)]
                bc_vc.connect(ba_vl)
                d_bc_vc.connect(ba_vl)
                self.split_edge(b_vu.x, ba_vl.x)
                yield ba_vl
                ba_vu = self.V[tuple(ba_vu)]
                bc_vc.connect(ba_vu)
                d_bc_vc.connect(ba_vu)
                os_v = self.split_edge(vectors[1].x, ba_vu.x)
                ss_v = self.split_edge(b_vl.x, ba_vu.x)
                yield os_v.x
                yield ss_v.x
                yield ba_vu
                d_bc_vc = self.split_edge(vectors[0].x, bc_vc.x)
                d_bc_vc.connect(vco)
                yield d_bc_vc.x
                d_b_vl = self.split_edge(vectors[1].x, b_vl.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_b_vl)
                yield d_b_vl.x
                d_b_vu = self.split_edge(vectors[2].x, b_vu.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_b_vu)
                yield d_b_vu.x
                d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_ba_vl)
                yield d_ba_vl
                d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)
                d_bc_vc.connect(vco)
                d_bc_vc.connect(d_ba_vu)
                yield d_ba_vu
                c_vc, vl, vu, a_vl, a_vu = vectors
                comb = [vl, vu, a_vl, a_vu, b_vl, b_vu, ba_vl, ba_vu]
                comb_iter = itertools.combinations(comb, 2)
                for vecs in comb_iter:
                    self.split_edge(vecs[0].x, vecs[1].x)
                ab_C.append((bc_vc, b_vl, b_vu, ba_vl, ba_vu))
                ab_C.append((d_bc_vc, d_b_vl, d_b_vu, d_ba_vl, d_ba_vu))
                ab_C.append((d_bc_vc, vectors[1], b_vl, a_vu, ba_vu))
                ab_C.append((d_bc_vc, vu, b_vu, a_vl, ba_vl))
            for j, (VL, VC, VU) in enumerate(zip(cCox, cCcx, cCux)):
                for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                    a_vl = list(vl.x)
                    a_vu = list(vu.x)
                    a_vl[i + 1] = vut[i + 1]
                    a_vu[i + 1] = vut[i + 1]
                    a_vl = self.V[tuple(a_vl)]
                    a_vu = self.V[tuple(a_vu)]
                    c_vc = self.split_edge(vl.x, a_vu.x)
                    self.split_edge(vl.x, vu.x)
                    c_vc.connect(vco)
                    c_vc.connect(vc)
                    c_vc.connect(vl)
                    c_vc.connect(vu)
                    c_vc.connect(a_vl)
                    c_vc.connect(a_vu)
                    yield c_vc.x
                    c_vl = self.split_edge(vl.x, a_vl.x)
                    c_vl.connect(vco)
                    c_vc.connect(c_vl)
                    yield c_vl.x
                    c_vu = self.split_edge(vu.x, a_vu.x)
                    c_vu.connect(vco)
                    c_vc.connect(c_vu)
                    yield c_vu.x
                    a_vc = self.split_edge(a_vl.x, a_vu.x)
                    a_vc.connect(vco)
                    a_vc.connect(c_vc)
                    ab_C.append((c_vc, vl, vu, a_vl, a_vu))
                    Cox[i + 1].append(vl)
                    Cox[i + 1].append(vc)
                    Cox[i + 1].append(vu)
                    Ccx[i + 1].append(c_vl)
                    Ccx[i + 1].append(c_vc)
                    Ccx[i + 1].append(c_vu)
                    Cux[i + 1].append(a_vl)
                    Cux[i + 1].append(a_vc)
                    Cux[i + 1].append(a_vu)
                    Cox[j].append(c_vl)
                    Cox[j].append(a_vl)
                    Ccx[j].append(c_vc)
                    Ccx[j].append(a_vc)
                    Cux[j].append(c_vu)
                    Cux[j].append(a_vu)
                    yield a_vc.x
        except IndexError:
            for vectors in ab_Cc:
                ba_vl = list(vectors[3].x)
                ba_vu = list(vectors[4].x)
                ba_vl[i + 1] = vut[i + 1]
                ba_vu[i + 1] = vut[i + 1]
                ba_vu = self.V[tuple(ba_vu)]
                yield ba_vu
                d_bc_vc = self.split_edge(vectors[1].x, ba_vu.x)
                yield ba_vu
                d_bc_vc.connect(vectors[1])
                d_bc_vc.connect(vectors[2])
                d_bc_vc.connect(vectors[3])
                d_bc_vc.connect(vectors[4])
                yield d_bc_vc.x
                ba_vl = self.V[tuple(ba_vl)]
                yield ba_vl
                d_ba_vl = self.split_edge(vectors[3].x, ba_vl.x)
                d_ba_vu = self.split_edge(vectors[4].x, ba_vu.x)
                d_ba_vc = self.split_edge(d_ba_vl.x, d_ba_vu.x)
                yield d_ba_vl
                yield d_ba_vu
                yield d_ba_vc
                c_vc, vl, vu, a_vl, a_vu = vectors
                comb = [vl, vu, a_vl, a_vu, ba_vl, ba_vu]
                comb_iter = itertools.combinations(comb, 2)
                for vecs in comb_iter:
                    self.split_edge(vecs[0].x, vecs[1].x)
            cCox = Cox[i]
            cCcx = Ccx[i]
            cCux = Cux[i]
            VL, VC, VU = (cCox, cCcx, cCux)
            for k, (vl, vc, vu) in enumerate(zip(VL, VC, VU)):
                a_vu = list(vu.x)
                a_vu[i + 1] = vut[i + 1]
                a_vu = self.V[tuple(a_vu)]
                yield a_vl.x
                c_vc = self.split_edge(vl.x, a_vu.x)
                self.split_edge(vl.x, vu.x)
                c_vc.connect(vco)
                c_vc.connect(vc)
                c_vc.connect(vl)
                c_vc.connect(vu)
                c_vc.connect(a_vu)
                yield c_vc.x
                c_vu = self.split_edge(vu.x, a_vu.x)
                c_vu.connect(vco)
                c_vc.connect(c_vu)
                yield c_vu.x
                Cox[i + 1].append(vu)
                Ccx[i + 1].append(c_vu)
                Cux[i + 1].append(a_vu)
                s_ab_C.append([c_vc, vl, vu, a_vu])
                yield a_vu.x
    try:
        del Cox
        del Ccx
        del Cux
        del ab_C
        del ab_Cc
    except UnboundLocalError:
        pass
    try:
        self.triangulated_vectors.remove((tuple(origin_c), tuple(supremum_c)))
    except ValueError:
        pass
    for vs in sup_set:
        self.triangulated_vectors.append((tuple(vco.x), tuple(vs.x)))
    if centroid:
        vcn_set = set()
        c_nn_lists = []
        for vs in sup_set:
            c_nn = self.vpool(vco.x, vs.x)
            try:
                c_nn.remove(vcn_set)
            except KeyError:
                pass
            c_nn_lists.append(c_nn)
        for c_nn in c_nn_lists:
            try:
                c_nn.remove(vcn_set)
            except KeyError:
                pass
        for vs, c_nn in zip(sup_set, c_nn_lists):
            vcn = self.split_edge(vco.x, vs.x)
            vcn_set.add(vcn)
            try:
                c_nn.remove(vcn_set)
            except KeyError:
                pass
            for vnn in c_nn:
                vcn.connect(vnn)
            yield vcn.x
    else:
        pass
    yield vut
    return