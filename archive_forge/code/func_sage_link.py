from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def sage_link(self):
    """
        Convert to a SageMath Knot or Link::

           sage: L = Link('K10n11')   # Spherogram link
           sage: K = L.sage_link(); K
           Knot represented by 10 crossings
           sage: L.alexander_polynomial()/K.alexander_polynomial()  # Agree up to units
           -t^3
           sage: L.signature(), K.signature()
           (-4, -4)

        Can also go the other way::

           sage: L = Link('K11n11')
           sage: M = Link(L.sage_link())
           sage: L.signature(), M.signature()
           (-2, -2)

        Can also take a braid group perspective.

            sage: B = BraidGroup(4)
            sage: a, b, c = B.gens()
            sage: Link(braid_closure=(a**-3) * (b**4) * (c**2) * a * b * c )
            <Link: 2 comp; 12 cross>
            sage: L = Link(a * b * c); L
            <Link: 1 comp; 3 cross>
            sage: S = L.sage_link(); S
            Knot represented by 3 crossings
            sage: Link(S)
            <Link: 1 comp; 3 cross>
        """
    if SageKnot is None:
        raise ValueError('Your SageMath does not seem to have a native link type')
    sage_type = SageKnot if len(self.link_components) == 1 else SageLink
    code = self.PD_code(min_strand_index=1)
    if sage_pd_clockwise:
        code = [[x[0], x[3], x[2], x[1]] for x in code]
    return sage_type(code)