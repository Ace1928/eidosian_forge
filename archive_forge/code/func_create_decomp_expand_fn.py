import contextlib
import pennylane as qml
from pennylane.operation import (
def create_decomp_expand_fn(custom_decomps, dev, decomp_depth=10):
    """Creates a custom expansion function for a device that applies
    a set of specified custom decompositions.

    Args:
        custom_decomps (Dict[Union(str, qml.operation.Operation), Callable]): Custom
            decompositions to be applied by the device at runtime.
        dev (pennylane.Device): A quantum device.
        decomp_depth: The maximum depth of the expansion.

    Returns:
        Callable: A custom expansion function that a device can call to expand
        its tapes within a context manager that applies custom decompositions.

    **Example**

    Suppose we would like a custom expansion function that decomposes all CNOTs
    into CZs. We first define a decomposition function:

    .. code-block:: python

        def custom_cnot(wires):
            return [
                qml.Hadamard(wires=wires[1]),
                qml.CZ(wires=[wires[0], wires[1]]),
                qml.Hadamard(wires=wires[1])
            ]

    We then create the custom function (passing a device, in order to pick up any
    additional stopping criteria the expansion should have), and then register the
    result as a custom function of the device:

    >>> custom_decomps = {qml.CNOT : custom_cnot}
    >>> expand_fn = qml.transforms.create_decomp_expand_fn(custom_decomps, dev)
    >>> dev.custom_expand(expand_fn)
    """
    custom_op_names = [op if isinstance(op, str) else op.__name__ for op in custom_decomps.keys()]
    custom_fn = qml.transforms.create_expand_fn(decomp_depth, stop_at=qml.BooleanFn(lambda obj: obj.name not in custom_op_names), device=dev)

    def custom_decomp_expand(self, circuit, max_expansion=decomp_depth):
        with _custom_decomp_context(custom_decomps):
            return custom_fn(circuit, max_expansion=max_expansion)
    return custom_decomp_expand