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
def findall_operations_until_blocked(self, start_frontier: Dict['cirq.Qid', int], is_blocker: Callable[['cirq.Operation'], bool]=lambda op: False) -> List[Tuple[int, 'cirq.Operation']]:
    """Finds all operations until a blocking operation is hit.

        An operation is considered blocking if both of the following hold:

        - It is in the 'light cone' of start_frontier.
        - `is_blocker` returns a truthy value, or it acts on a blocked qubit

        Every qubit acted on by a blocking operation is thereafter itself
        blocked.

        The notion of reachability here differs from that in
        reachable_frontier_from in two respects:

        - An operation is not considered blocking only because it is in a
            moment before the start_frontier of one of the qubits on which it
            acts.
        - Operations that act on qubits not in start_frontier are not
            automatically blocking.

        For every (moment_index, operation) returned:

        - moment_index >= min((start_frontier[q] for q in operation.qubits
            if q in start_frontier), default=0)
        - set(operation.qubits).intersection(start_frontier)

        Below are some examples, where on the left the opening parentheses show
        `start_frontier` and on the right are the operations included (with
        their moment indices) in the output. `F` and `T` indicate that
        `is_blocker` return `False` or `True`, respectively, when applied to
        the gates; `M` indicates that it doesn't matter.

        ```
            ─(─F───F───────    ┄(─F───F─)┄┄┄┄┄
               │   │              │   │
            ─(─F───F───T─── => ┄(─F───F─)┄┄┄┄┄
                       │                  ┊
            ───────────T───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─────(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
               │       │          ┊       │
            ───M───M─(─F───    ┄┄┄┄┄┄┄┄┄(─F─)┄┄
                   │        =>        ┊
            ───────M───M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                  ┊
            ───────────M───    ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ───M─(─────M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
               │       │           ┊       ┊
            ───M─(─T───M───     ┄┄┄┄┄()┄┄┄┄┄┄┄┄
                   │        =>         ┊
            ───────T───M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄
                       │                   ┊
            ───────────M───     ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄


            ─(─F───F───    ┄(─F───F─)┄
               │   │    =>    │   │
            ───F─(─F───    ┄(─F───F─)┄


            ─(─F───────────    ┄(─F─)┄┄┄┄┄┄┄┄┄
               │                  │
            ───F───F───────    ┄(─F─)┄┄┄┄┄┄┄┄┄
                   │        =>        ┊
            ───────F───F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
                       │                  │
            ─(─────────F───    ┄┄┄┄┄┄┄┄┄(─F─)┄
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            A list of tuples. Each tuple describes an operation found between
            the start frontier and a blocking operation. The first item of
            each tuple is the index of the moment containing the operation,
            and the second item is the operation itself.

        """
    op_list: List[Tuple[int, ops.Operation]] = []
    if not start_frontier:
        return op_list
    start_index = min(start_frontier.values())
    blocked_qubits: Set[cirq.Qid] = set()
    for index, moment in enumerate(self[start_index:], start_index):
        active_qubits = set((q for q, s in start_frontier.items() if s <= index))
        for op in moment.operations:
            if is_blocker(op) or blocked_qubits.intersection(op.qubits):
                blocked_qubits.update(op.qubits)
            elif active_qubits.intersection(op.qubits):
                op_list.append((index, op))
        if blocked_qubits.issuperset(start_frontier):
            break
    return op_list