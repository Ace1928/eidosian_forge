import abc
from typing import (
from cirq import ops, value, devices
def modify_column(self, column: List[Optional['Cell']]) -> None:
    """Applies this cell's modification to its column.

        For example, a control cell will add a control qubit to other operations
        in the column.

        Args:
            column: A mutable list of cells in the column, including empty
                cells (with value `None`). This method is permitted to change
                the items in the list, but must not change the length of the
                list.

        Returns:
            Nothing. The `column` argument is mutated in place.
        """