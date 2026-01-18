from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator, Stinespring
from .dihedral.random import random_cnotdihedral
from .symplectic.random import random_clifford, random_pauli, random_pauli_list
def random_quantum_channel(input_dims: int | tuple | None=None, output_dims: int | tuple | None=None, rank: int | None=None, seed: int | np.random.Generator | None=None):
    """Return a random CPTP quantum channel.

    This constructs the Stinespring operator for the quantum channel by
    sampling a random isometry from the unitary Haar measure.

    Args:
        input_dims (int or tuple): the input dimension of the channel.
        output_dims (int or tuple): the input dimension of the channel.
        rank (int): Optional. The rank of the quantum channel Choi-matrix.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Stinespring: a quantum channel operator.

    Raises:
        QiskitError: if rank or dimensions are invalid.
    """
    if input_dims is None and output_dims is None:
        raise QiskitError('No dimensions specified: input_dims and output_dims cannot both be None.')
    if input_dims is None:
        input_dims = output_dims
    elif output_dims is None:
        output_dims = input_dims
    d_in = np.prod(input_dims)
    d_out = np.prod(output_dims)
    if rank is None or rank > d_in * d_out:
        rank = d_in * d_out
    if rank < 1:
        raise QiskitError(f'Rank {rank} must be greater than 0.')
    from scipy import stats
    unitary = stats.unitary_group.rvs(max(rank * d_out, d_in), random_state=seed)
    return Stinespring(unitary[:, :d_in], input_dims=input_dims, output_dims=output_dims)