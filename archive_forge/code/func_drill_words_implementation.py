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
def drill_words_implementation(manifold, words, verified, bits_prec, perturb=False, verbose: bool=False):
    mcomplex = Mcomplex(manifold)
    add_r13_geometry(mcomplex, manifold, verified=verified, bits_prec=bits_prec)
    geodesics: Sequence[GeodesicInfo] = [compute_geodesic_info(mcomplex, word) for word in words]
    index_geodesics_and_add_post_drill_infos(geodesics, mcomplex)
    geodesics_to_drill = [g for g in geodesics if not g.core_curve_cusp]
    if perturb:
        perturb_geodesics(mcomplex, geodesics_to_drill, verbose=verbose)
    drilled_mcomplex: Mcomplex = drill_geodesics(mcomplex, geodesics_to_drill, verbose=verbose)
    post_drill_infos: Sequence[CuspPostDrillInfo] = reorder_vertices_and_get_post_drill_infos(drilled_mcomplex)
    drilled_manifold = drilled_mcomplex.snappy_manifold()
    refill_and_adjust_peripheral_curves(drilled_manifold, post_drill_infos)
    drilled_manifold.set_name(manifold.name() + '_drilled')
    return drilled_manifold