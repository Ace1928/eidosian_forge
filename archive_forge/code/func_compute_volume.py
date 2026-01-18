from ..sage_helper import sage_method, _within_sage
from ..number import Number
from . import verifyHyperbolicity
def compute_volume(manifold, verified, bits_prec=None):
    """
    Computes the volume of the given manifold. If verified is used,
    the hyperbolicity is checked rigorously and the volume is given as
    verified interval.

    >>> M = Manifold('m004')
    >>> vol = M.volume(bits_prec=100)
    >>> vol # doctest: +ELLIPSIS
    2.029883212819307250042405108...

    sage: ver_vol = M.volume(verified=True)
    sage: vol in ver_vol
    True
    sage: 2.02988321283 in ver_vol
    False
    """
    shape_intervals = manifold.tetrahedra_shapes('rect', bits_prec=bits_prec, intervals=verified)
    if verified:
        verifyHyperbolicity.check_logarithmic_gluing_equations_and_positively_oriented_tets(manifold, shape_intervals)
    volume = sum([_volume_from_shape(shape_interval) for shape_interval in shape_intervals])
    if isinstance(volume, Number):
        volume = manifold._number_(volume)
    return volume