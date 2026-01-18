import numpy as np
from pennylane.fermi import FermiSentence, FermiWord
from .observable_hf import qubit_observable
def spinz(orbitals):
    """Computes the total spin projection observable :math:`\\hat{S}_z`.

    The total spin projection operator :math:`\\hat{S}_z` is given by

    .. math::

        \\hat{S}_z = \\sum_{\\alpha, \\beta} \\langle \\alpha \\vert \\hat{s}_z \\vert \\beta \\rangle
        ~ \\hat{c}_\\alpha^\\dagger \\hat{c}_\\beta, ~~ \\langle \\alpha \\vert \\hat{s}_z
        \\vert \\beta \\rangle = s_{z_\\alpha} \\delta_{\\alpha,\\beta},

    where :math:`s_{z_\\alpha} = \\pm 1/2` is the spin-projection of the single-particle state
    :math:`\\vert \\alpha \\rangle`. The operators :math:`\\hat{c}^\\dagger` and :math:`\\hat{c}`
    are the particle creation and annihilation operators, respectively.

    Args:
        orbitals (str): Number of *spin* orbitals. If an active space is defined, this is
            the number of active spin-orbitals.

    Returns:
        pennylane.Hamiltonian: the total spin projection observable :math:`\\hat{S}_z`

    Raises:
        ValueError: If orbitals is less than or equal to 0

    **Example**

    >>> orbitals = 4
    >>> print(spinz(orbitals))
    (-0.25) [Z0]
    + (0.25) [Z1]
    + (-0.25) [Z2]
    + (0.25) [Z3]
    """
    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")
    r = np.arange(orbitals)
    sz_orb = np.where(r % 2 == 0, 0.5, -0.5)
    table = np.vstack([r, r, sz_orb]).T
    sentence = FermiSentence({})
    for i in table:
        sentence.update({FermiWord({(0, int(i[0])): '+', (1, int(i[1])): '-'}): i[2]})
    sentence.simplify()
    return qubit_observable(sentence)