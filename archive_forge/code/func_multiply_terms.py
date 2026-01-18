from ..sage_helper import _within_sage
from ..pari import Gen, pari
from ..math_basics import prod
def multiply_terms(self):
    """
        Multiplies all the terms that make up the numerator and denominator
        so that there is at most one term in each.

        >>> nf = pari("x^97+x^3+x+32121")
        >>> a = pari("43*x^3 + 1")
        >>> b = pari("x^2 + 3")
        >>> c = pari("x^2 + 4")

        >>> r1 = RUR.from_pari_fraction_and_number_field(a / b, nf)
        >>> r2 = RUR.from_pari_fraction_and_number_field(c, nf)
        >>> r = r1 * r2

        >>> r
        ( Mod(43*x^3 + 1, x^97 + x^3 + x + 32121) * Mod(x^2 + 4, x^97 + x^3 + x + 32121) ) / ( Mod(x^2 + 3, x^97 + x^3 + x + 32121) )

        >>> r.multiply_terms()
        ( Mod(43*x^5 + 172*x^3 + x^2 + 4, x^97 + x^3 + x + 32121) ) / ( Mod(x^2 + 3, x^97 + x^3 + x + 32121) )

        """
    return RUR([(self._numerator(), 1), (self._denominator(), -1)])