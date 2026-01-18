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
def reachable_frontier_from(self, start_frontier: Dict['cirq.Qid', int], *, is_blocker: Callable[['cirq.Operation'], bool]=lambda op: False) -> Dict['cirq.Qid', int]:
    """Determines how far can be reached into a circuit under certain rules.

        The location L = (qubit, moment_index) is *reachable* if and only if the
        following all hold true:

        - There is not a blocking operation covering L.
        -  At least one of the following holds:
            - qubit is in start frontier and moment_index =
                max(start_frontier[qubit], 0).
            - There is no operation at L and prev(L) = (qubit,
                moment_index-1) is reachable.
            - There is an (non-blocking) operation P covering L such that
                (q', moment_index - 1) is reachable for every q' on which P
                acts.

        An operation in moment moment_index is blocking if at least one of the
        following hold:

        - `is_blocker` returns a truthy value.
        - The operation acts on a qubit not in start_frontier.
        - The operation acts on a qubit q such that start_frontier[q] >
            moment_index.

        In other words, the reachable region extends forward through time along
        each qubit in start_frontier until it hits a blocking operation. Any
        location involving a qubit not in start_frontier is unreachable.

        For each qubit q in `start_frontier`, the reachable locations will
        correspond to a contiguous range starting at start_frontier[q] and
        ending just before some index end_q. The result of this method is a
        dictionary, and that dictionary maps each qubit q to its end_q.

        Examples:

        If `start_frontier` is

        ```
        {
            cirq.LineQubit(0): 6,
            cirq.LineQubit(1): 2,
            cirq.LineQubit(2): 2
        }
        ```

        then the reachable wire locations in the following circuit are
        highlighted with '█' characters:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ───H───@─────────────────█████████████████████─@───H───
                      │                                       │
            1: ───────@─██H███@██████████████████████─@───H───@───────
                              │                       │
            2: ─────────██████@███H██─@───────@───H───@───────────────
                                      │       │
            3: ───────────────────────@───H───@───────────────────────
        ```

        And the computed `end_frontier` is

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 9,
            cirq.LineQubit(2): 6,
        }
        ```

        Note that the frontier indices (shown above the circuit) are
        best thought of (and shown) as happening *between* moment indices.

        If we specify a blocker as follows:

        ```
        is_blocker=lambda: op == cirq.CZ(cirq.LineQubit(1),
                                         cirq.LineQubit(2))
        ```

        and use this `start_frontier`:

        ```
        {
            cirq.LineQubit(0): 0,
            cirq.LineQubit(1): 0,
            cirq.LineQubit(2): 0,
            cirq.LineQubit(3): 0,
        }
        ```

        Then this is the reachable area:

        ```

                0   1   2   3   4   5   6   7   8   9   10  11  12  13
            0: ─██H███@██████████████████████████████████████─@───H───
                      │                                       │
            1: ─██████@███H██─@───────────────────────@───H───@───────
                              │                       │
            2: ─█████████████─@───H───@───────@───H───@───────────────
                                      │       │
            3: ─█████████████████████─@───H───@───────────────────────

        ```

        and the computed `end_frontier` is:

        ```
        {
            cirq.LineQubit(0): 11,
            cirq.LineQubit(1): 3,
            cirq.LineQubit(2): 3,
            cirq.LineQubit(3): 5,
        }
        ```

        Args:
            start_frontier: A starting set of reachable locations.
            is_blocker: A predicate that determines if operations block
                reachability. Any location covered by an operation that causes
                `is_blocker` to return True is considered to be an unreachable
                location.

        Returns:
            An end_frontier dictionary, containing an end index for each qubit q
            mapped to a start index by the given `start_frontier` dictionary.

            To determine if a location (q, i) was reachable, you can use
            this expression:

                q in start_frontier and start_frontier[q] <= i < end_frontier[q]

            where i is the moment index, q is the qubit, and end_frontier is the
            result of this method.
        """
    active: Set['cirq.Qid'] = set()
    end_frontier = {}
    queue = BucketPriorityQueue[ops.Operation](drop_duplicate_entries=True)

    def enqueue_next(qubit: 'cirq.Qid', moment: int) -> None:
        next_moment = self.next_moment_operating_on([qubit], moment)
        if next_moment is None:
            end_frontier[qubit] = max(len(self), start_frontier[qubit])
            if qubit in active:
                active.remove(qubit)
        else:
            next_op = self.operation_at(qubit, next_moment)
            assert next_op is not None
            queue.enqueue(next_moment, next_op)
    for start_qubit, start_moment in start_frontier.items():
        enqueue_next(start_qubit, start_moment)
    while queue:
        cur_moment, cur_op = queue.dequeue()
        for q in cur_op.qubits:
            if q in start_frontier and cur_moment >= start_frontier[q] and (q not in end_frontier):
                active.add(q)
        continue_past = cur_op is not None and active.issuperset(cur_op.qubits) and (not is_blocker(cur_op))
        if continue_past:
            for q in cur_op.qubits:
                enqueue_next(q, cur_moment + 1)
        else:
            for q in cur_op.qubits:
                if q in active:
                    end_frontier[q] = cur_moment
                    active.remove(q)
    return end_frontier