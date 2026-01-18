from . import links_base, alexander
from .links_base import CrossingStrand, Crossing
from ..sage_helper import _within_sage, sage_method, sage_pd_clockwise
@sage_method
def morse_diagram(self):
    """
        Returns a MorseLinkDiagram of this link diagram, that is a choice
        of height function which realizes the Morse number::

            sage: L = Link('L8n2')
            sage: D = L.morse_diagram()
            sage: D.morse_number == L.morse_number()
            True
            sage: D.is_bridge()
            True
            sage: B = D.bridge()
            sage: len(B.bohua_code())
            64
        """
    from . import morse
    return morse.MorseLinkDiagram(self)