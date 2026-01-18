import pennylane as qml
from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable
def diff_hamiltonian(mol, cutoff=1e-12, core=None, active=None):
    """Return a function that computes the qubit Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the qubit hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = diff_hamiltonian(mol)(*args)
    >>> h.coeffs
    array([ 0.29817879+0.j,  0.20813365+0.j,  0.20813365+0.j,
             0.17860977+0.j,  0.04256036+0.j, -0.04256036+0.j,
            -0.04256036+0.j,  0.04256036+0.j, -0.34724873+0.j,
             0.13290293+0.j, -0.34724873+0.j,  0.17546329+0.j,
             0.17546329+0.j,  0.13290293+0.j,  0.18470917+0.j])
    """

    def _molecular_hamiltonian(*args):
        """Compute the qubit hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            Hamiltonian: the qubit Hamiltonian
        """
        h_ferm = fermionic_hamiltonian(mol, cutoff, core, active)(*args)
        return qubit_observable(h_ferm)
    return _molecular_hamiltonian