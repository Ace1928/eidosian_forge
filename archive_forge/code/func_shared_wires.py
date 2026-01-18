import functools
import itertools
from collections.abc import Iterable, Sequence
import numpy as np
from pennylane.pytrees import register_pytree
@staticmethod
def shared_wires(list_of_wires):
    """Return only the wires that appear in each Wires object in the list.

        This is similar to a set intersection method, but keeps the order of wires as they appear in the list.

        Args:
            list_of_wires (List[Wires]): list of Wires objects

        Returns:
            Wires: shared wires

        **Example**

        >>> wires1 =  Wires([4, 0, 1])
        >>> wires2 = Wires([3, 0, 4])
        >>> wires3 = Wires([4, 0])
        >>> Wires.shared_wires([wires1, wires2, wires3])
        <Wires = [4, 0]>
        >>> Wires.shared_wires([wires2, wires1, wires3])
        <Wires = [0, 4]>
        """
    for wires in list_of_wires:
        if not isinstance(wires, Wires):
            raise WireError(f'Expected a Wires object; got {wires} of type {type(wires)}.')
    sets_of_wires = [wire.toset() for wire in list_of_wires]
    intersecting_wires = functools.reduce(lambda a, b: a & b, sets_of_wires)
    shared = []
    for wire in list_of_wires[0]:
        if wire in intersecting_wires:
            shared.append(wire)
    return Wires(tuple(shared), _override=True)