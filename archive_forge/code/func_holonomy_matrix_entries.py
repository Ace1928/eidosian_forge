from .shapes import polished_tetrahedra_shapes
from ..sage_helper import _within_sage, sage_method
from .polished_reps import polished_holonomy
from . import nsagetools, interval_reps, slice_obs_HKL
from .character_varieties import character_variety, character_variety_ideal
@sage_method
def holonomy_matrix_entries(manifold, fundamental_group_args=[], match_kernel=True):
    """
    The entries of the matrices of the holonomy as list of ApproximateAlgebraicNumbers
    (four consecutive numbers per matrix). The numbers are guaranteed to lie in the
    trace field only if match_kernel = False::

        sage: M = Manifold("m004")
        sage: mat_entries = M.holonomy_matrix_entries(match_kernel = False) # doctest: +NORMALIZE_WHITESPACE +NUMERIC9
        sage: mat_entries
        <SetOfAAN: [0.5 + 0.8660254037844386*I, 0.5 - 0.8660254037844386*I, 0.5 + 0.8660254037844386*I, 1.0 - 1.7320508075688772*I, 1.0 - 3.4641016151377544*I, -2.0 + 1.7320508075688772*I, -1.0 - 1.7320508075688772*I, 1.7320508075688772*I]>
        sage: K = mat_entries.find_field(100, 10, optimize = True)[0]
        sage: K.polynomial()
        x^2 - x + 1
    """

    def func(prec):
        G = polished_holonomy(manifold, prec, fundamental_group_args=fundamental_group_args, match_kernel=match_kernel)
        return sum([G.SL2C(g).list() for g in G.generators()], [])
    return ListOfApproximateAlgebraicNumbers(func)