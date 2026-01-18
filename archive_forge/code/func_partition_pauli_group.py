from functools import lru_cache, reduce, singledispatch
from itertools import product
from typing import List, Union
from warnings import warn
import numpy as np
import pennylane as qml
from pennylane.operation import Tensor
from pennylane.ops import Hamiltonian, Identity, PauliX, PauliY, PauliZ, Prod, SProd
from pennylane.tape import OperationRecorder
from pennylane.wires import Wires
@lru_cache()
def partition_pauli_group(n_qubits: int) -> List[List[str]]:
    """Partitions the :math:`n`-qubit Pauli group into qubit-wise commuting terms.

    The :math:`n`-qubit Pauli group is composed of :math:`4^{n}` terms that can be partitioned into
    :math:`3^{n}` qubit-wise commuting groups.

    Args:
        n_qubits (int): number of qubits

    Returns:
        List[List[str]]: A collection of qubit-wise commuting groups containing Pauli words as
        strings

    **Example**

    >>> qml.pauli.partition_pauli_group(3)
    [['III', 'IIZ', 'IZI', 'IZZ', 'ZII', 'ZIZ', 'ZZI', 'ZZZ'],
     ['IIX', 'IZX', 'ZIX', 'ZZX'],
     ['IIY', 'IZY', 'ZIY', 'ZZY'],
     ['IXI', 'IXZ', 'ZXI', 'ZXZ'],
     ['IXX', 'ZXX'],
     ['IXY', 'ZXY'],
     ['IYI', 'IYZ', 'ZYI', 'ZYZ'],
     ['IYX', 'ZYX'],
     ['IYY', 'ZYY'],
     ['XII', 'XIZ', 'XZI', 'XZZ'],
     ['XIX', 'XZX'],
     ['XIY', 'XZY'],
     ['XXI', 'XXZ'],
     ['XXX'],
     ['XXY'],
     ['XYI', 'XYZ'],
     ['XYX'],
     ['XYY'],
     ['YII', 'YIZ', 'YZI', 'YZZ'],
     ['YIX', 'YZX'],
     ['YIY', 'YZY'],
     ['YXI', 'YXZ'],
     ['YXX'],
     ['YXY'],
     ['YYI', 'YYZ'],
     ['YYX'],
     ['YYY']]
    """
    if isinstance(n_qubits, float):
        if n_qubits.is_integer():
            n_qubits = int(n_qubits)
    if not isinstance(n_qubits, int):
        raise TypeError('Must specify an integer number of qubits.')
    if n_qubits < 0:
        raise ValueError('Number of qubits must be at least 0.')
    if n_qubits == 0:
        return [['']]
    strings = set()
    groups = []
    for string in product('FXYZ', repeat=n_qubits):
        if string not in strings:
            num_free_slots = string.count('F')
            group = []
            commuting = product('IZ', repeat=num_free_slots)
            for commuting_string in commuting:
                commuting_string = list(commuting_string)
                new_string = tuple((commuting_string.pop(0) if s == 'F' else s for s in string))
                if new_string not in strings:
                    group.append(''.join(new_string))
                    strings |= {new_string}
            if len(group) > 0:
                groups.append(group)
    return groups