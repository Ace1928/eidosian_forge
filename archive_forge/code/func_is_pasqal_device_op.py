from typing import Sequence, Any, Union, Dict
import numpy as np
import networkx as nx
import cirq
from cirq import GridQubit, LineQubit
from cirq.ops import NamedQubit
from cirq_pasqal import ThreeDQubit, TwoDQubit, PasqalGateset
def is_pasqal_device_op(self, op: cirq.Operation) -> bool:
    if not isinstance(op, cirq.Operation):
        raise ValueError('Got unknown operation:', op)
    return op in self.gateset