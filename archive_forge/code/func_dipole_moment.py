import pennylane as qml
from pennylane.fermi import FermiSentence, FermiWord
from .basis_data import atomic_numbers
from .hartree_fock import scf
from .matrices import moment_matrix
from .observable_hf import fermionic_observable, qubit_observable
def dipole_moment(mol, cutoff=1e-16, core=None, active=None):
    """Return a function that computes the qubit dipole moment observable.

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

    The fermonic dipole operator is then transformed to the qubit basis which gives

    .. math::

        \\hat{D} = \\sum_{j} c_j P_j,

    where :math:`c_j` is a numerical coefficient and :math:`P_j` is a ternsor product of
    single-qubit Pauli operators :math:`X, Y, Z, I`.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible dipole moment integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the qubit dipole moment observable

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> dipole_moment(mol)(*args)[2].ops
    [I(0),
     Z(0),
     Y(0) @ Z(1) @ Y(2),
     X(0) @ Z(1) @ X(2),
     Z(1),
     Y(1) @ Z(2) @ Y(3),
     X(1) @ Z(2) @ X(3),
     Z(2),
     Z(3)]
    """

    def _dipole(*args):
        """Compute the qubit dipole moment observable.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            (list[Hamiltonian]): x, y and z components of the dipole moment observable
        """
        d = []
        d_ferm = fermionic_dipole(mol, cutoff, core, active)(*args)
        for i in d_ferm:
            d.append(qubit_observable(i, cutoff=cutoff))
        return d
    return _dipole