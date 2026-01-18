from typing import Callable, Iterator, Sequence, Tuple, TYPE_CHECKING
from cirq import ops, value
from cirq.interop.quirk.cells.cell import CELL_SIZES, CellMaker
def generate_all_qubit_permutation_cell_makers() -> Iterator[CellMaker]:
    yield from _permutation_family('<<', 'left_rotate', lambda _, x: x + 1)
    yield from _permutation_family('>>', 'right_rotate', lambda _, x: x - 1)
    yield from _permutation_family('rev', 'reverse', lambda _, x: ~x)
    yield from _permutation_family('weave', 'interleave', _interleave_bit)
    yield from _permutation_family('split', 'deinterleave', _deinterleave_bit)