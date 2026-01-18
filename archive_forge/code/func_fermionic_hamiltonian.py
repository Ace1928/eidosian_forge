import pennylane as qml
from .hartree_fock import nuclear_energy, scf
from .observable_hf import fermionic_observable, qubit_observable
def fermionic_hamiltonian(mol, cutoff=1e-12, core=None, active=None):
    """Return a function that computes the fermionic Hamiltonian.

    Args:
        mol (~qchem.molecule.Molecule): the molecule object
        cutoff (float): cutoff value for discarding the negligible electronic integrals
        core (list[int]): indices of the core orbitals
        active (list[int]): indices of the active orbitals

    Returns:
        function: function that computes the fermionic hamiltonian

    **Example**

    >>> symbols  = ['H', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad = False)
    >>> alpha = np.array([[3.42525091, 0.62391373, 0.1688554],
    >>>                   [3.42525091, 0.62391373, 0.1688554]], requires_grad=True)
    >>> mol = qml.qchem.Molecule(symbols, geometry, alpha=alpha)
    >>> args = [alpha]
    >>> h = fermionic_hamiltonian(mol)(*args)
    """

    def _fermionic_hamiltonian(*args):
        """Compute the fermionic hamiltonian.

        Args:
            *args (array[array[float]]): initial values of the differentiable parameters

        Returns:
            FermiSentence: fermionic Hamiltonian
        """
        core_constant, one, two = electron_integrals(mol, core, active)(*args)
        return fermionic_observable(core_constant, one, two, cutoff)
    return _fermionic_hamiltonian