from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def morse_number(self, solver='GLPK'):
    """
        The *Morse number* of a planar link diagram D is

            m(D) = min { # of maxima of h on D }

        where h is a height function on R^2 which is generic on D; alternatively,
        this is the minimum number of cups/caps in a `MorseLink presentation
        <http://katlas.math.toronto.edu/wiki/MorseLink_Presentations>`_
        of the diagram D.  The Morse number is very closely related to the more
        traditional bridge number.   Examples::

            sage: K = Link('5_2')
            sage: K.morse_number()
            2
            sage: Link('6^3_2').morse_number()
            3
        """
    from . import morse
    return morse.morse_via_LP(self, solver)[0]