def mass_from_composition(composition):
    """Calculates molecular mass from atomic weights

    Parameters
    ----------
    composition: dict
        Dictionary mapping int (atomic number) to int (coefficient)

    Returns
    -------
    float
        molecular weight in atomic mass units


    Notes
    -----
    Atomic number 0 denotes charge or "net electron defficiency"

    Examples
    --------
    >>> '%.2f' % mass_from_composition({0: -1, 1: 1, 8: 1})
    '17.01'
    """
    mass = 0.0
    for k, v in composition.items():
        if k == 0:
            mass -= v * 0.0005489
        else:
            mass += v * relative_atomic_masses[k - 1]
    return mass