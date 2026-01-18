import numpy as np
import warnings
from ..util import NoConvergence
def density_from_concentration(conc, T=None, molar_mass=None, rho_cb=sulfuric_acid_density, units=None, atol=None, maxiter=10, warn=False, **kwargs):
    """Calculates the density of a solution from its concentration

    Given a function which calculates the density of a solution from the mass
    fraction of the solute, this function calculates (iteratively) the density
    of said solution for a given concentration.

    Parameters
    ----------
    conc : float (optionally with units)
        Concentration (mol / m³).
    T : float (optionally with units)
        Passed to ``rho_cb``.
    molar_mass : float (optionally with units)
        Molar mass of solute.
    rho_cb : callback
        Callback with signature f(w, T, units=None) -> rho
        (default: :func:`sulfuric_acid_density`).
    units : object (optional)
        Object with attributes: meter, kilogram, mol.
    atol : float (optionally with units)
        Convergence criterion for fixed-point iteration
        (default: 1e-3 kg/m³).
    maxiter : int
        Maximum number of iterations (when exceeded a NoConvergence exception
        is raised).
    \\*\\*kwargs:
        Keyword arguments passed onto ``rho_cb``.

    Returns
    -------
    Density of sulfuric acid (float of kg/m³ if T is float and units is None)

    Examples
    --------
    >>> print('%d' % density_from_concentration(400, 293))
    1021

    Raises
    ------
    chempy.util.NoConvergence:
        When maxiter is exceeded

    """
    if units is None:
        m = 1
        kg = 1
        mol = 1
    else:
        m = units.meter
        kg = units.kilogram
        mol = units.mol
    kg_per_m3 = kg * m ** (-3)
    if atol is None:
        atol = 0.001 * kg_per_m3
    if molar_mass is None:
        molar_mass = (1.00794 * 2 + 32.066 + 4 * 15.9994) * 0.001 * kg / mol
    if units is not None:
        conc = conc.rescale(mol / m ** 3)
        molar_mass = molar_mass.rescale(kg / mol)
    rho = 1100 * kg_per_m3
    delta_rho = float('inf') * kg_per_m3
    iter_idx = 0
    while atol < abs(delta_rho):
        new_rho = rho_cb(conc * molar_mass / rho, T, units=units, warn=warn, **kwargs)
        delta_rho = new_rho - rho
        rho = new_rho
        iter_idx += 1
        if iter_idx > maxiter:
            raise NoConvergence('maxiter exceeded')
    return rho