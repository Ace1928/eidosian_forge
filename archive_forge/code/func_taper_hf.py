import functools
import itertools
import numpy as np
import scipy
import pennylane as qml
from pennylane.operation import active_new_opmath
from pennylane.pauli import PauliSentence, PauliWord, pauli_sentence, simplify
from pennylane.pauli.utils import _binary_matrix_from_pws
from pennylane.wires import Wires
def taper_hf(generators, paulixops, paulix_sector, num_electrons, num_wires):
    """Transform a Hartree-Fock state with a Clifford operator and then taper qubits.

    The fermionic operators defining the molecule's Hartree-Fock (HF) state are first mapped onto a qubit operator
    using the Jordan-Wigner encoding. This operator is then transformed using the Clifford operators :math:`U`
    obtained from the :math:`\\mathbb{Z}_2` symmetries of the molecular Hamiltonian resulting in a qubit operator
    that acts non-trivially only on a subset of qubits. A new, tapered HF state is built on this reduced subset
    of qubits by placing the qubits which are acted on by a Pauli-X or Pauli-Y operators in state :math:`|1\\rangle`
    and leaving the rest in state :math:`|0\\rangle`.

    Args:
        generators (list[Operator]): list of generators of symmetries, taus, for the Hamiltonian
        paulixops (list[Operation]):  list of single-qubit Pauli-X operators
        paulix_sector (list[int]): list of eigenvalues of Pauli-X operators
        num_electrons (int): number of active electrons in the system
        num_wires (int): number of wires in the system for generating the Hartree-Fock bitstring

    Returns:
        array(int): tapered Hartree-Fock state

    **Example**

    >>> symbols = ['He', 'H']
    >>> geometry = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4588684632]])
    >>> mol = qml.qchem.Molecule(symbols, geometry, charge=1)
    >>> H, n_qubits = qml.qchem.molecular_hamiltonian(symbols, geometry, charge=1)
    >>> n_elec = mol.n_electrons
    >>> generators = qml.qchem.symmetry_generators(H)
    >>> paulixops = qml.qchem.paulix_ops(generators, 4)
    >>> paulix_sector = qml.qchem.optimal_sector(H, generators, n_elec)
    >>> taper_hf(generators, paulixops, paulix_sector, n_elec, n_qubits)
    tensor([1, 1], requires_grad=True)
    """
    hf = np.where(np.arange(num_wires) < num_electrons, 1, 0)
    ferm_ps = PauliSentence({PauliWord({0: 'I'}): 1.0})
    for idx, bit in enumerate(hf):
        if bit:
            ps = qml.jordan_wigner(qml.FermiC(idx), ps=True)
        else:
            ps = PauliSentence({PauliWord({idx: 'I'}): 1.0})
        ferm_ps @= ps
    fermop_taper = _taper_pauli_sentence(ferm_ps, generators, paulixops, paulix_sector)
    fermop_ps = pauli_sentence(fermop_taper)
    fermop_mat = _binary_matrix_from_pws(list(fermop_ps), len(fermop_taper.wires))
    gen_wires = Wires.all_wires([generator.wires for generator in generators])
    xop_wires = Wires.all_wires([paulix_op.wires for paulix_op in paulixops])
    wireset = Wires.unique_wires([gen_wires, xop_wires])
    tapered_hartree_fock = []
    for col in fermop_mat.T[fermop_mat.shape[1] // 2:]:
        if 1 in col:
            tapered_hartree_fock.append(1)
        else:
            tapered_hartree_fock.append(0)
    while len(tapered_hartree_fock) < len(wireset):
        tapered_hartree_fock.append(0)
    return np.array(tapered_hartree_fock).astype(int)