import sys
from . import links, tangles
def some_knots():
    from . import hyperbolic_montesinos
    return [(K, knot(fractions)) for K, fractions in hyperbolic_montesinos.knots]