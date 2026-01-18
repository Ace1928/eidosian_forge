import warnings
import pennylane as qml
from pennylane.operation import Operation, AnyWires
def compute_indices_MPS(wires, n_block_wires, offset=None):
    """Generate a list containing the wires for each block.

    Args:
        wires (Iterable): wires that the template acts on
        n_block_wires (int): number of wires per block_gen
        offset (int): offset value for positioning the subsequent blocks relative to each other.
            If ``None``, it defaults to :math:`\\text{offset} = \\lfloor \\text{n_block_wires}/2  \\rfloor`,
            otherwise :math:`\\text{offset} \\in [1, \\text{n_block_wires} - 1]`.

    Returns:
        layers (Tuple[Tuple]]): array of wire indices or wire labels for each block
    """
    n_wires = len(wires)
    if n_block_wires < 2:
        raise ValueError(f'The number of wires in each block must be larger than or equal to 2; got n_block_wires = {n_block_wires}')
    if n_block_wires > n_wires:
        raise ValueError(f'n_block_wires must be smaller than or equal to the number of wires; got n_block_wires = {n_block_wires} and number of wires = {n_wires}')
    if offset is None:
        offset = n_block_wires // 2
    if offset < 1 or offset > n_block_wires - 1:
        raise ValueError(f'Provided offset is outside the expected range; the expected range for n_block_wires = {n_block_wires} is range{(1, n_block_wires - 1)}')
    n_step = offset
    n_layers = len(wires) - int(len(wires) % (n_block_wires // 2)) - n_step
    return tuple((tuple((wires[idx] for idx in range(j, j + n_block_wires))) for j in range(0, n_layers, n_step) if not j + n_block_wires > len(wires)))