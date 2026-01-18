from typing import Callable, Iterable, Iterator, NoReturn, Union, TYPE_CHECKING
from typing_extensions import Protocol
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops.raw_types import Operation
def freeze_op_tree(root: OP_TREE) -> OP_TREE:
    """Replaces all iterables in the OP_TREE with tuples.

    Args:
        root: The operation or tree of operations to freeze.

    Returns:
        An OP_TREE with the same operations and branching structure, but where
        all internal nodes are tuples instead of arbitrary iterables.
    """
    return transform_op_tree(root, iter_transformation=tuple)