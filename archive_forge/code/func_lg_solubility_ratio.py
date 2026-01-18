import warnings
def lg_solubility_ratio(electrolytes, gas, units=None, warn=True):
    """Returns the log10 value of the solubilty ratio

    Implements equation 16, p 156. from Schumpe (1993)

    Parameters
    ----------
    electrolytes : dict
        Mapping substance key (one in ``p_ion_rM``) to concentration.
    gas : str
        Substance key for the gas (one in ``p_gas_rM``).
    units : object (optional)
        object with attribute: molar
    warn : bool (default: True)
        Emit UserWarning when 'F-' among electrolytes.

    """
    if units is None:
        M = 1
    else:
        M = units.molar
    if warn and 'F-' in electrolytes:
        warnings.warn('In Schumpe 1993: data for fluoride uncertain.')
    return sum([(p_gas_rM[gas] / M + p_ion_rM[k] / M) * v for k, v in electrolytes.items()])