from typing import Callable
from cirq import circuits, devices, ops
def map_moment(self, moment: circuits.Moment) -> circuits.Moment:
    return circuits.Moment((self.map_operation(op) for op in moment.operations))