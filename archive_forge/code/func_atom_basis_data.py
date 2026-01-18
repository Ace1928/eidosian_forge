from .basis_data import basis_sets, load_basisset
def atom_basis_data(name, atom, load_data=False):
    """Generate default basis set parameters for an atom.

    This function extracts the angular momentum, exponents, and contraction coefficients of
    Gaussian functions forming atomic orbitals for a given atom. These values are taken, by default,
    from the basis set data provided in :mod:`~pennylane.qchem.basis_data`. If `load_data = True`,
    the basis set data is loaded from the basis-set-exchange library.

    Args:
        name (str): name of the basis set
        atom (str): atomic symbol of the chemical element
        load_data (bool): flag to load data from the basis-set-exchange library

    Returns:
        list(tuple): tuple containing the angular momentum, the exponents and contraction
        coefficients of a basis function

    **Example**

    >>> params = atom_basis_data('sto-3g', 'H')
    >>> print(params)
    [((0, 0, 0), [3.425250914, 0.6239137298, 0.168855404], [0.1543289673, 0.5353281423, 0.4446345422])]
    """
    name = name.lower()
    s = [(0, 0, 0)]
    p = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    d = [(1, 1, 0), (1, 0, 1), (0, 1, 1), (2, 0, 0), (0, 2, 0), (0, 0, 2)]
    if load_data:
        basis = load_basisset(name, atom)
    else:
        basis = basis_sets[name][atom]
    params = []
    sp_count = 0
    for i, j in enumerate(basis['orbitals']):
        if j == 'S':
            params.append((s[0], basis['exponents'][i], basis['coefficients'][i]))
        if j == 'SP':
            for term in j:
                if term == 'S':
                    params.append((s[0], basis['exponents'][i], basis['coefficients'][i + sp_count]))
                if term == 'P':
                    for l in p:
                        params.append((l, basis['exponents'][i], basis['coefficients'][i + sp_count + 1]))
            sp_count += 1
        if j == 'P':
            for l in p:
                params.append((l, basis['exponents'][i], basis['coefficients'][i]))
        if j == 'D':
            for l in d:
                params.append((l, basis['exponents'][i], basis['coefficients'][i]))
    return params