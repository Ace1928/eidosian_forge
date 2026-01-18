import numpy as np
from pennylane.fermi import FermiSentence, FermiWord
from .observable_hf import qubit_observable
def spin2(electrons, orbitals):
    """Compute the total spin observable :math:`\\hat{S}^2`.

    The total spin observable :math:`\\hat{S}^2` is given by

    .. math::

        \\hat{S}^2 = \\frac{3}{4}N + \\sum_{ \\bm{\\alpha}, \\bm{\\beta}, \\bm{\\gamma}, \\bm{\\delta} }
        \\langle \\bm{\\alpha}, \\bm{\\beta} \\vert \\hat{s}_1 \\cdot \\hat{s}_2
        \\vert \\bm{\\gamma}, \\bm{\\delta} \\rangle ~
        \\hat{c}_\\bm{\\alpha}^\\dagger \\hat{c}_\\bm{\\beta}^\\dagger
        \\hat{c}_\\bm{\\gamma} \\hat{c}_\\bm{\\delta},

    where the two-particle matrix elements are computed as,

    .. math::

        \\langle \\bm{\\alpha}, \\bm{\\beta} \\vert \\hat{s}_1 \\cdot \\hat{s}_2
        \\vert \\bm{\\gamma}, \\bm{\\delta} \\rangle = && \\delta_{\\alpha,\\delta} \\delta_{\\beta,\\gamma} \\\\
        && \\times \\left( \\frac{1}{2} \\delta_{s_{z_\\alpha}, s_{z_\\delta}+1}
        \\delta_{s_{z_\\beta}, s_{z_\\gamma}-1} + \\frac{1}{2} \\delta_{s_{z_\\alpha}, s_{z_\\delta}-1}
        \\delta_{s_{z_\\beta}, s_{z_\\gamma}+1} + s_{z_\\alpha} s_{z_\\beta}
        \\delta_{s_{z_\\alpha}, s_{z_\\delta}} \\delta_{s_{z_\\beta}, s_{z_\\gamma}} \\right).

    In the equations above :math:`N` is the number of electrons, :math:`\\alpha` refer to the
    quantum numbers of the spatial wave function and :math:`s_{z_\\alpha}` is
    the spin projection of the single-particle state
    :math:`\\vert \\bm{\\alpha} \\rangle \\equiv \\vert \\alpha, s_{z_\\alpha} \\rangle`.
    The operators :math:`\\hat{c}^\\dagger` and :math:`\\hat{c}` are the particle creation
    and annihilation operators, respectively.

    Args:
        electrons (int): Number of electrons. If an active space is defined, this is
            the number of active electrons.
        orbitals (int): Number of *spin* orbitals. If an active space is defined,  this is
            the number of active spin-orbitals.

    Returns:
        pennylane.Hamiltonian: the total spin observable :math:`\\hat{S}^2`

    Raises:
        ValueError: If electrons or orbitals is less than or equal to 0

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> print(spin2(electrons, orbitals))
    (0.75) [I0]
    + (0.375) [Z1]
    + (-0.375) [Z0 Z1]
    + (0.125) [Z0 Z2]
    + (0.375) [Z0]
    + (-0.125) [Z0 Z3]
    + (-0.125) [Z1 Z2]
    + (0.125) [Z1 Z3]
    + (0.375) [Z2]
    + (0.375) [Z3]
    + (-0.375) [Z2 Z3]
    + (0.125) [Y0 X1 Y2 X3]
    + (0.125) [Y0 Y1 X2 X3]
    + (0.125) [Y0 Y1 Y2 Y3]
    + (-0.125) [Y0 X1 X2 Y3]
    + (-0.125) [X0 Y1 Y2 X3]
    + (0.125) [X0 X1 X2 X3]
    + (0.125) [X0 X1 Y2 Y3]
    + (0.125) [X0 Y1 X2 Y3]
    """
    if electrons <= 0:
        raise ValueError(f"'electrons' must be greater than 0; got for 'electrons' {electrons}")
    if orbitals <= 0:
        raise ValueError(f"'orbitals' must be greater than 0; got for 'orbitals' {orbitals}")
    sz = np.where(np.arange(orbitals) % 2 == 0, 0.5, -0.5)
    table = _spin2_matrix_elements(sz)
    sentence = FermiSentence({FermiWord({}): 3 / 4 * electrons})
    for i in table:
        sentence.update({FermiWord({(0, int(i[0])): '+', (1, int(i[1])): '+', (2, int(i[2])): '-', (3, int(i[3])): '-'}): i[4]})
    sentence.simplify()
    return qubit_observable(sentence)