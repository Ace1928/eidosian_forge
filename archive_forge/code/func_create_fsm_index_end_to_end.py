from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def create_fsm_index_end_to_end(fsm_info: FSMInfo, vocabulary: Dict[str, List[int]]) -> Dict[int, Set[Tuple[int, int]]]:
    """Create an FSM state-to-vocabulary map/index through end-to-end token parsing."""
    states_to_token_subsets: Dict[int, Set[Tuple[int, int]]] = {}
    seen: Set[int] = set()
    next_states = {fsm_info.initial}
    while next_states:
        start_state = next_states.pop()
        token_ids_end_states = state_scan_tokens(fsm_info.transitions, fsm_info.alphabet_symbol_mapping, fsm_info.alphabet_anything_value, fsm_info.initial, fsm_info.finals, vocabulary, start_state)
        for token_id_and_end_state in token_ids_end_states:
            states_to_token_subsets.setdefault(start_state, set()).add(token_id_and_end_state)
            end_state = token_id_and_end_state[1]
            if end_state not in seen:
                next_states.add(end_state)
        seen.add(start_state)
    return states_to_token_subsets