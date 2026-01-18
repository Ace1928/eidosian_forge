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
def dummy_function_for_additional_doctests():
    """
    Test with manifold without symmetry. Note that the code in drilling is
    deterministic but the SnapPea kernel code to remove the finite vertices
    and simplify is not. Thus, we need canonical_retriangulation() to get
    a consistent result:

        >>> from snappy import Manifold, ManifoldHP
        >>> from snappy.drilling.exceptions import GeodesicSystemNotSimpleError
        >>> M = Manifold("v2986")
        >>> M.drill_word('gB').canonical_retriangulation().triangulation_isosig(ignore_orientation=False)
        'kLvvAQQkbhijhghgjijxxacvcccccv_baBaaBDbBa'

    Test non-simple geodesic and verified computation:

        sage: M = ManifoldHP("m004")
        sage: try:
        ...       M.drill_word('bbCC', verified = True)
        ... except GeodesicSystemNotSimpleError as e:
        ...     print("Not simple")
        Not simple

    Tests drilling one geodesic that intersects 1-skeleton::

        >>> M = Manifold("m125")
        >>> M.drill_word('d').triangulation_isosig(ignore_orientation=False)
        'gLLPQcdefeffpvauppb_acbBbBaaBbacbBa'

    Tests drilling two geodesics that intersect each other:

        >>> try: # doctest: +NUMERIC9
        ...     M.drill_words(['d','Ad'])
        ... except GeodesicSystemNotSimpleError as e:
        ...     print("Max tube radius:", e.maximal_tube_radius)
        Max tube radius: 0.0000000000

    Tests drilling geodesics that are entirely in the 2-skeleton::

        >>> M.drill_words(['a','acAADa']).triangulation_isosig(ignore_orientation=False)
        'iLMvPQcbbdfhgghhpuabpauab_acbdaBbaBbaBcBBbcbbb'

    Same test as verified computation::

        sage: M.drill_words(['a','acAADa'], verified = True).triangulation_isosig(ignore_orientation=False)
        'iLMvPQcbbdfhgghhpuabpauab_acbdaBbaBbaBcBBbcbbb'

    Test error when drilling something close to core curve::

        >>> from snappy import Manifold
        >>> M = Manifold("m125")
        >>> MM = M.drill_word('d')
        >>> MM.dehn_fill((1,0),2)
        >>> bad_word = 'bc'
        >>> MM.drill_word(bad_word) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        snappy.drilling.exceptions.GeodesicCloseToCoreCurve: The given geodesic is very close to a core curve and might intersect it.

    There are two places where we detect whether the geodesic is close
    to a core curve (rather than tiling forever). Test the other place
    in the GeodesicTube code used to determine the maximal amount we can
    perturb the geodesic:

        >>> drill_words_implementation(MM, [bad_word], verified = False, bits_prec = 53, perturb = True) # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        snappy.drilling.exceptions.GeodesicCloseToCoreCurve: The given geodesic is very close to a core curve and might intersect it.

    A particular tricky case in terms testing that the start piece is correctly
    handled by 2-3 moves (in particular, commit f9879d04 introduced a bug):

        >>> Manifold("m004").drill_words(['CAC','CCbC']).canonical_retriangulation().triangulation_isosig(ignore_orientation=False)
        'qLvvLvAMQQQkcgimopkllmpkonnnpixcaelchapewetvrn_bcaaBbBBbaBaBbB'


    An interesting case where geodesic intersects triangulation in only one tetrahedron:

        >>> Manifold("m019").drill_word('A').canonical_retriangulation().triangulation_isosig(ignore_orientation=False)
        'gLLPQccdefffqffqqof_BaaBdcbb'

    A bug in an earlier implementation found by Nathan Dunfield (where putting the words in one order caused a failure):

        >>> import sys
        >>> original_limit = sys.getrecursionlimit()
        >>> sys.setrecursionlimit(100000)
        >>> def drilled_isosig(M, words):
        ...     for i in range(10):
        ...         try:
        ...             F = M.drill_words(words).filled_triangulation()
        ...             return F.canonical_retriangulation().triangulation_isosig(ignore_orientation=False)
        ...         except RuntimeError:
        ...             pass
        >>> drilled_isosig(Manifold('K11n34(0,1)'), ['iFcdbEiFJ', 'iFJ'])
        'zLLvLLwzAwPQMQzzQkcdgijkjplssrnrotqruvwyxyxyhsgnnighueqdniblsipklpxgcr_BcaBbBba'
        >>> drilled_isosig(Manifold('K11n34(0,1)'), ['iFJ', 'iFcdbEiFJ'])
        'zLLvLLwzAwPQMQzzQkcdgijkjplssrnrotqruvwyxyxyhsgnnighueqdniblsipklpxgcr_babBbaBcaB'
        >>> sys.setrecursionlimit(original_limit)

    """