from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
@numba.njit(cache=True, nogil=True)
def state_scan_tokens(fsm_transitions: Dict[Tuple[int, int], int], alphabet_symbol_mapping: Dict[str, int], alphabet_anything_value: int, fsm_initial: int, fsm_finals: Set[int], vocabulary: Dict[str, List[int]], start_state: int) -> Set[Tuple[int, int]]:
    res = set()
    for token, token_ids in vocabulary.items():
        state_seq = _walk_fsm(fsm_transitions, alphabet_symbol_mapping, alphabet_anything_value, fsm_initial, fsm_finals, token, start_state, False)
        if state_seq is not None and len(state_seq) < len(token):
            continue
        for token_id in token_ids:
            res.add((token_id, state_seq[-1]))
    return res