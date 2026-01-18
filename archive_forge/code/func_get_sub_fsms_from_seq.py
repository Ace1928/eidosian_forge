from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def get_sub_fsms_from_seq(state_seq: Sequence[int], fsms_to_trans_finals: Dict[int, Tuple[Set[Tuple[int, int]], Set[int], Dict[int, Set[int]]]]) -> Generator[Tuple[int, bool, bool], None, None]:
    """Get the indices of the sub-FSMs in `fsm` that could have matched the state sequence `state_seq`.

    Parameters
    ----------
    state_seq
        A state sequence.
    fsms_to_trans_finals
        A map from FSM indices to tuples containing sets of their state transitions
        and sets of the final/accept states.

    Returns
    -------
    A generator returning tuples containing each sub-FSM index (in the order
    they were union-ed to construct `fsm`) and booleans indicating whether or
    not there is another valid transition from the last state in the sequence
    for the associated sub-FSM (i.e. if the FSM can continue
    accepting/matching) and whether or not the sequence ends in a final state
    of the sub-FSM.
    """
    state_seq_transitions = set(zip(state_seq[:-1], state_seq[1:]))
    last_fsm_state = state_seq[-1]
    yield from ((fsm_idx, any((last_fsm_state == from_s for from_s, to_s in transitions)), state_seq[-1] in finals) for fsm_idx, (transitions, finals, _) in fsms_to_trans_finals.items() if state_seq_transitions.issubset(transitions))