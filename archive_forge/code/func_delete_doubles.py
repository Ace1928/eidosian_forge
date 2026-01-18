from .cartan_type import CartanType
from mpmath import fac
from sympy.core.backend import Matrix, eye, Rational, igcd
from sympy.core.basic import Atom
def delete_doubles(self, reflections):
    """
        This is a helper method for determining the order of an element in the
        Weyl group of G2.  It takes a Weyl element and if repeated simple reflections
        in it, it deletes them.
        """
    counter = 0
    copy = list(reflections)
    for elt in copy:
        if counter < len(copy) - 1:
            if copy[counter + 1] == elt:
                del copy[counter]
                del copy[counter]
        counter += 1
    return copy