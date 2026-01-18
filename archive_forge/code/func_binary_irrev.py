from .._util import get_backend
def binary_irrev(t, kf, prod, major, minor, backend=None):
    """Analytic product transient of a irreversible 2-to-1 reaction.

    Product concentration vs time from second order irreversible kinetics.

    Parameters
    ----------
    t : float, Symbol or array_like
    kf : number or Symbol
        Forward (bimolecular) rate constant.
    prod : number or Symbol
        Initial concentration of the complex.
    major : number or Symbol
        Initial concentration of the more abundant reactant.
    minor : number or Symbol
        Initial concentration of the less abundant reactant.
    backend : module or str
        Default is 'numpy', can also be e.g. ``sympy``.

    """
    be = get_backend(backend)
    return prod + major * (1 - be.exp(-kf * (major - minor) * t)) / (major / minor - be.exp(-kf * t * (major - minor)))