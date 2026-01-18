from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
def knot_floer_homology(self, prime=2, complex=False):
    """
        Uses Zoltán Szabó's HFK Calculator to compute the knot Floer
        homology.  This also gives the Seifert genus, whether the knot
        fibers, etc:

        >>> K = Link('K3a1')
        >>> K.knot_floer_homology()    # doctest: +NORMALIZE_WHITESPACE
        {'L_space_knot': True,
         'epsilon': 1,
         'fibered': True,
         'modulus': 2,
         'nu': 1,
         'ranks': {(-1, -2): 1, (0, -1): 1, (1, 0): 1},
         'seifert_genus': 1,
         'tau': 1,
         'total_rank': 3}

        The homology itself is encoded by 'ranks', with the form::

          (Alexander grading, Maslov grading): dimension

        For example, here is the Conway knot, which has Alexander
        polynomial 1 and genus 3:

        >>> L = Link('K11n34')
        >>> ranks = L.knot_floer_homology()['ranks']
        >>> [(a, m) for a, m in ranks if a == 3]
        [(3, 3), (3, 4)]
        >>> ranks[3, 3], ranks[3, 4]
        (1, 1)

        Computation is done over F_2 by default, other primes less
        than 2^15 can be used instead via the optional "prime"
        parameter.

        If the parameter `complex` is set to True, then the simplified
        "UV = 0" knot Floer chain complex is returned. This complex is
        computed over the ring F[U,V]/(UV = 0), where F is the integers
        mod the chosen prime; this corresponds to only the horizontal and
        vertical arrows in the full knot Floer complex. The complex is
        specified by:

        * generators: a dictionary from the generator names to their
          (Alexander, Maslov) gradings.  The number of generators is
          equal to the total_rank.

        * differential: a dictionary whose value on (a, b) is an integer
          specifying the coefficient on the differential from generator a
          to generator b, where only nonzero differentials are
          recorded. (The coefficient on the differential is really an
          element of F[U,V]/(UV = 0), but the power of U or V can be
          recovered from the gradings on a and b so only the element of F
          is recorded.)

        For example, to compute the vertical differential, whose homology
        is HFhat(S^3), you can do:

        >>> data = L.knot_floer_homology(prime=31991, complex=True)
        >>> gens, diff = data['generators'], data['differentials']
        >>> vert = {(i,j):diff[i, j] for i, j in diff
        ...                          if gens[i][1] == gens[j][1] + 1}

        sage: from sage.all import matrix, GF
        sage: M = matrix(GF(31991), len(gens), len(gens), vert, sparse=True)
        sage: M*M == 0
        True
        sage: M.right_kernel().rank() - M.rank()
        1
        """
    import knot_floer_homology
    if len(self.link_components) + self.unlinked_unknot_components > 1:
        raise ValueError('Only works for knots, this has more components')
    if len(self.link_components) == 0 and self.unlinked_unknot_components == 1:
        return Link(braid_closure=[1, 1, -1]).knot_floer_homology()
    return knot_floer_homology.pd_to_hfk(self, prime=prime, complex=complex)