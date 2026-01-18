from .cusps import CuspPostDrillInfo
from .geometric_structure import compute_r13_planes_for_tet
from .tracing import compute_plane_intersection_param, Endpoint, GeodesicPiece
from .epsilons import compute_epsilon
from . import constants
from . import exceptions
from ..snap.t3mlite import simplex, Perm4, Tetrahedron # type: ignore
from ..matrix import matrix # type: ignore
from ..exceptions import InsufficientPrecisionError # type: ignore
from typing import Sequence, Optional, Union, Tuple, List, Dict, Mapping
def one_four_move(given_pieces: Sequence[GeodesicPiece], verified: bool) -> Sequence[GeodesicPiece]:
    tet: Tetrahedron = given_pieces[0].tet
    RF = tet.O13_matrices[simplex.F0].base_ring()
    if not given_pieces[0].endpoints[0].subsimplex in simplex.TwoSubsimplices:
        raise Exception('Expected given geodesic piece to start on a face for one-four move.')
    if not given_pieces[-1].endpoints[1].subsimplex in simplex.TwoSubsimplices:
        raise Exception('Expected given geodesic piece to end on a face for one-four move.')
    n = len(given_pieces)
    if n == 1:
        bias = RF(constants.piece_midpoint_bias)
        new_point = given_pieces[0].endpoints[0].r13_point + bias * given_pieces[0].endpoints[1].r13_point
    elif n == 2:
        if not given_pieces[0].endpoints[1].subsimplex == simplex.T:
            raise Exception('Expected middle point to be in the tetrahedron when given two pieces to one-four move.')
        if not given_pieces[1].endpoints[0].subsimplex == simplex.T:
            raise Exception('Expected middle point to be in the tetrahedron when given two pieces to one-four move.')
        if not given_pieces[0].tet is given_pieces[1].tet:
            raise Exception('Expected pieces to be in the same tetrahedron when given two pieces to one-four move.')
        new_point = given_pieces[0].endpoints[1].r13_point
    else:
        raise Exception('Bad 1-4 move')
    new_tets: dict[int, Tetrahedron] = {f: Tetrahedron() for f in simplex.TwoSubsimplices}
    neighbors: dict[int, Tetrahedron] = {f: t for f, t in tet.Neighbor.items()}
    gluings: dict[int, Perm4] = {f: p for f, p in tet.Gluing.items()}
    id_matrix = matrix.identity(ring=RF, n=4)
    for f0, new_tet0 in new_tets.items():
        new_tet0.geodesic_pieces = []
        v0 = simplex.comp(f0)
        new_tet0.R13_vertices = {v0: new_point}
        new_tet0.post_drill_infos = {v0: CuspPostDrillInfo(index=given_pieces[0].index)}
        new_tet0.O13_matrices = {}
        new_tet0.PeripheralCurves = [[{v: {face: 0 for face in simplex.TwoSubsimplices} for v in simplex.ZeroSubsimplices} for sheet in range(2)] for ml in range(2)]
        for f1, new_tet1 in new_tets.items():
            if f0 != f1:
                new_tet0.attach(f1, new_tet1, _swap_perms[f0, f1])
                v1 = simplex.comp(f1)
                new_tet0.R13_vertices[v1] = tet.R13_vertices[v1]
                new_tet0.post_drill_infos[v1] = tet.post_drill_infos[v1]
                new_tet0.O13_matrices[f1] = id_matrix
        neighbor: Tetrahedron = neighbors[f0]
        gluing: Perm4 = gluings[f0]
        if neighbor is tet:
            other_tet = new_tets[gluing.image(f0)]
        else:
            other_tet = neighbor
        new_tet0.attach(f0, other_tet, gluing.tuple())
        new_tet0.O13_matrices[f0] = tet.O13_matrices[f0]
        compute_r13_planes_for_tet(new_tet0)
    for ml in range(2):
        for sheet in range(2):
            for v, faces in simplex.FacesAroundVertexCounterclockwise.items():
                for face in faces:
                    new_tets[face].PeripheralCurves[ml][sheet][v][face] = tet.PeripheralCurves[ml][sheet][v][face]
                f0, f1, f2 = faces
                new_tets[f0].PeripheralCurves[ml][sheet][v][f1] = -tet.PeripheralCurves[ml][sheet][v][f0]
                new_tets[f1].PeripheralCurves[ml][sheet][v][f0] = tet.PeripheralCurves[ml][sheet][v][f0]
                new_tets[f1].PeripheralCurves[ml][sheet][v][f2] = tet.PeripheralCurves[ml][sheet][v][f2]
                new_tets[f2].PeripheralCurves[ml][sheet][v][f1] = -tet.PeripheralCurves[ml][sheet][v][f2]
    for old_piece in tet.geodesic_pieces:
        if old_piece in given_pieces:
            continue
        start_subsimplex: int = old_piece.endpoints[0].subsimplex
        end_subsimplex: int = old_piece.endpoints[1].subsimplex
        total_subsimplex = start_subsimplex | end_subsimplex
        if total_subsimplex in simplex.OneSubsimplices:
            for face, new_tet in new_tets.items():
                if simplex.is_subset(total_subsimplex, face):
                    GeodesicPiece.replace_by(old_piece, old_piece, [GeodesicPiece.create_and_attach(old_piece.index, new_tet, old_piece.endpoints)])
                    break
            else:
                raise Exception('Unhandled edge case')
            continue
        r13_endpoints = [e.r13_point for e in old_piece.endpoints]
        retrace_direction: int = +1
        end_cell_dimension: int = 2
        if end_subsimplex in simplex.ZeroSubsimplices:
            end_cell_dimension = 0
            allowed_end_corners: Sequence[Tuple[Tetrahedron, int]] = [(new_tets[f], end_subsimplex) for f in simplex.TwoSubsimplices if simplex.is_subset(end_subsimplex, f)]
        elif end_subsimplex == simplex.T:
            end_cell_dimension = 3
            allowed_end_corners = [(new_tet, end_subsimplex) for new_tet in new_tets.values()]
        elif start_subsimplex == simplex.T:
            end_cell_dimension = 3
            retrace_direction = -1
            start_subsimplex, end_subsimplex = (end_subsimplex, start_subsimplex)
            r13_endpoints = r13_endpoints[::-1]
            allowed_end_corners = [(new_tet, end_subsimplex) for new_tet in new_tets.values()]
        elif start_subsimplex in simplex.TwoSubsimplices and end_subsimplex in simplex.TwoSubsimplices:
            allowed_end_corners = [(new_tets[end_subsimplex], end_subsimplex)]
        else:
            raise Exception('Unhandled case')
        GeodesicPiece.replace_by(old_piece, old_piece, _retrace_geodesic_piece(old_piece.index, new_tets, new_tets[start_subsimplex], start_subsimplex, end_cell_dimension, r13_endpoints, retrace_direction, verified, allowed_end_corners=allowed_end_corners))
    start_point: Endpoint = given_pieces[0].endpoints[0]
    end_point: Endpoint = given_pieces[-1].endpoints[1]
    new_pieces: Sequence[GeodesicPiece] = [GeodesicPiece.create_face_to_vertex_and_attach(given_pieces[0].index, new_tets[start_point.subsimplex], start_point, direction=+1), GeodesicPiece.create_face_to_vertex_and_attach(given_pieces[0].index, new_tets[end_point.subsimplex], end_point, direction=-1)]
    GeodesicPiece.replace_by(given_pieces[0], given_pieces[-1], new_pieces)
    return new_pieces