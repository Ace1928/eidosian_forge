import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from .basis_data import atomic_numbers
from .hartree_fock import scf
from .matrices import moment_matrix
from .observable_hf import fermionic_observable, qubit_observable
def fermionic_dipole(mol, cutoff=1e-18, core=None, active=None):
    """Return a function that builds the fermionic dipole moment observable.

    The dipole operator in the second-quantized form is

    .. math::

        \\hat{D} = -\\sum_{pq} d_{pq} [\\hat{c}_{p\\uparrow}^\\dagger \\hat{c}_{q\\uparrow} +
        \\hat{c}_{p\\downarrow}^\\dagger \\hat{c}_{q\\downarrow}] -
        \\hat{D}_\\mathrm{c} + \\hat{D}_\\mathrm{n},

    where the matrix elements :math:`d_{pq}` are given by the integral of the position operator
    :math:`\\hat{{\\bf r}}` over molecular orbitals :math:`\\phi`

    .. math::

        d_{pq} = \\int \\phi_p^*(r) \\hat{{\\bf r}} \\phi_q(r) dr,

    and :math:`\\hat{c}^{\\dagger}` and :math:`\\hat{c}` are the creation and annihilation operators,
    respectively. The contribution of the core orbitals and nuclei are denoted by
    :math:`\\hat{D}_\\mathrm{c}` and :math:`\\hat{D}_\\mathrm{n}`, respectively, which are computed as

    .. math::

        \\hat{D}_\\mathrm{c} = 2 \\sum_{i=1}^{N_\\mathrm{core}} d_{ii},

    and

    .. math::

        \\hat{D}_\\mathrm{n} = \\sum_{i=1}^{N_\\mathrm{atoms}} Z_i {\\bf R}_i,

    where :math:`Z_i` and :math:`{\\bf R}_i` denote, respectively, the atomic number and the
    nuclear coordinates of the :math:`i`-th atom of the molecule.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible dipole moment integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that builds the fermionic dipole moment observable

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> fermionic_dipole(mol)(*args)[2]
    -0.4999999988651487 * a⁺(0) a(0)
    + 0.82709948984052 * a⁺(0) a(2)
    + -0.4999999988651487 * a⁺(1) a(1)
    + 0.82709948984052 * a⁺(1) a(3)
    + 0.82709948984052 * a⁺(2) a(0)
    + -0.4999999899792451 * a⁺(2) a(2)
    + 0.82709948984052 * a⁺(3) a(1)
    + -0.4999999899792451 * a⁺(3) a(3)
    + 1.0 * I
    """

    def _fermionic_dipole(*args):
        """Build the fermionic dipole moment observable.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic dipole moment
        """
        constants, integrals = dipole_integrals(mol, core, active)(*args)
        nd = [qml.math.array([0]), qml.math.array([0]), qml.math.array([0])]
        for i, s in enumerate(mol.symbols):
            nd[0] = nd[0] + atomic_numbers[s] * mol.coordinates[i][0]
            nd[1] = nd[1] + atomic_numbers[s] * mol.coordinates[i][1]
            nd[2] = nd[2] + atomic_numbers[s] * mol.coordinates[i][2]
        d_ferm = []
        for i in range(3):
            f = fermionic_observable(constants[i], integrals[i], cutoff=cutoff)
            d_ferm.append(FermiSentence({FermiWord({}): nd[i][0]}) - f)
        return d_ferm
    return _fermionic_dipole