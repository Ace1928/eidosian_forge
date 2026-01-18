import abc
import enum
import html
import itertools
import math
from collections import defaultdict
from typing import (
from typing_extensions import Self
import networkx
import numpy as np
import cirq._version
from cirq import _compat, devices, ops, protocols, qis
from cirq._doc import document
from cirq.circuits._bucket_priority_queue import BucketPriorityQueue
from cirq.circuits.circuit_operation import CircuitOperation
from cirq.circuits.insert_strategy import InsertStrategy
from cirq.circuits.qasm_output import QasmOutput
from cirq.circuits.text_diagram_drawer import TextDiagramDrawer
from cirq.circuits.moment import Moment
from cirq.protocols import circuit_diagram_info_protocol
from cirq.type_workarounds import NotImplementedType
def get_earliest_accommodating_moment_index(moment_or_operation: Union['cirq.Moment', 'cirq.Operation'], qubit_indices: Dict['cirq.Qid', int], mkey_indices: Dict['cirq.MeasurementKey', int], ckey_indices: Dict['cirq.MeasurementKey', int], length: Optional[int]=None) -> int:
    """Get the index of the earliest moment that can accommodate the given moment or operation.

    Updates the dictionaries keeping track of the last moment index addressing a given qubit,
    measurement key, and control key.

    Args:
        moment_or_operation: The moment operation in question.
        qubit_indices: A dictionary mapping qubits to the latest moments that address them.
        mkey_indices: A dictionary mapping measureent keys to the latest moments that address them.
        ckey_indices: A dictionary mapping control keys to the latest moments that address them.
        length: The length of the circuit that we are trying to insert a moment or operation into.
            Should probably be equal to the maximum of the values in `qubit_indices`,
            `mkey_indices`, and `ckey_indices`.

    Returns:
        The integer index of the earliest moment that can accommodate the given moment or operation.
    """
    mop_qubits = moment_or_operation.qubits
    mop_mkeys = protocols.measurement_key_objs(moment_or_operation)
    mop_ckeys = protocols.control_keys(moment_or_operation)
    if isinstance(moment_or_operation, Moment):
        if length is not None:
            last_conflict = length - 1
        else:
            last_conflict = max([*qubit_indices.values(), *mkey_indices.values(), *ckey_indices.values(), -1])
    else:
        last_conflict = -1
        if mop_qubits:
            last_conflict = max(last_conflict, *[qubit_indices.get(qubit, -1) for qubit in mop_qubits])
        if mop_mkeys:
            last_conflict = max(last_conflict, *[mkey_indices.get(key, -1) for key in mop_mkeys])
            last_conflict = max(last_conflict, *[ckey_indices.get(key, -1) for key in mop_mkeys])
        if mop_ckeys:
            last_conflict = max(last_conflict, *[mkey_indices.get(key, -1) for key in mop_ckeys])
    mop_index = last_conflict + 1
    for qubit in mop_qubits:
        qubit_indices[qubit] = mop_index
    for key in mop_mkeys:
        mkey_indices[key] = mop_index
    for key in mop_ckeys:
        ckey_indices[key] = mop_index
    return mop_index