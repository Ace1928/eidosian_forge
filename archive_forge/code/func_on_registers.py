import enum
import abc
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union, overload, Iterator
from numpy.typing import NDArray
import attr
import cirq
import numpy as np
from cirq_ft.deprecation import deprecated_cirq_ft_class
def on_registers(self, **qubit_regs: Union[cirq.Qid, Sequence[cirq.Qid], NDArray[cirq.Qid]]) -> cirq.Operation:
    return self.on(*merge_qubits(self.signature, **qubit_regs))