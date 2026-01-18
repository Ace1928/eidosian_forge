from sympy.utilities.iterables import sift
from .util import new
def rm_id(isid, new=new):
    """ Create a rule to remove identities.

    isid - fn :: x -> Bool  --- whether or not this element is an identity.

    Examples
    ========

    >>> from sympy.strategies import rm_id
    >>> from sympy import Basic, S
    >>> remove_zeros = rm_id(lambda x: x==0)
    >>> remove_zeros(Basic(S(1), S(0), S(2)))
    Basic(1, 2)
    >>> remove_zeros(Basic(S(0), S(0))) # If only identites then we keep one
    Basic(0)

    See Also:
        unpack
    """

    def ident_remove(expr):
        """ Remove identities """
        ids = list(map(isid, expr.args))
        if sum(ids) == 0:
            return expr
        elif sum(ids) != len(ids):
            return new(expr.__class__, *[arg for arg, x in zip(expr.args, ids) if not x])
        else:
            return new(expr.__class__, expr.args[0])
    return ident_remove