from . import exceptions
from . import epsilons
from . import debug
from .tracing import trace_geodesic
from .crush import crush_geodesic_pieces
from .line import R13LineWithMatrix
from .geometric_structure import add_r13_geometry, word_to_psl2c_matrix
from .geodesic_info import GeodesicInfo, sample_line
from .perturb import perturb_geodesics
from .subdivide import traverse_geodesics_to_subdivide
from .cusps import (
from ..snap.t3mlite import Mcomplex
from ..exceptions import InsufficientPrecisionError
import functools
from typing import Sequence
def drill_geodesics(mcomplex: Mcomplex, geodesics: Sequence[GeodesicInfo], verbose: bool=False) -> Mcomplex:
    """
    Given a triangulation with geometric structure attached with
    add_r13_geometry and basic information about geodesics, computes
    the triangulation (with finite vertices) obtained by drilling
    the geodesics.

    Each provided GeodesicInfo is supposed to have a start point and
    a tetrahedron in the fundamental domain that contains the start point
    in its interior and an end point such that the line segment from the
    start to the endpoint forms a closed curve in the manifold.
    """
    if len(geodesics) == 0:
        return mcomplex
    for g in geodesics:
        if not g.tet:
            raise exceptions.GeodesicStartPointOnTwoSkeletonError()
    all_pieces: Sequence[Sequence[GeodesicPiece]] = [trace_geodesic(g, verified=mcomplex.verified) for g in geodesics]
    if verbose:
        print('Number of geodesic pieces:', [len(pieces) for pieces in all_pieces])
    tetrahedra: Sequence[Tetrahedron] = traverse_geodesics_to_subdivide(mcomplex, all_pieces)
    if verbose:
        print('Number of tets after subdividing: %d' % len(tetrahedra))
    result: Mcomplex = crush_geodesic_pieces(tetrahedra)
    debug.check_vertex_indices(result.Tetrahedra)
    debug.check_peripheral_curves(result.Tetrahedra)
    return result