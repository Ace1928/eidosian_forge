from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def make_deterministic_fsm(fsm: FSM) -> Tuple[BetterFSM, Dict[int, int]]:
    """Construct an equivalent FSM with deterministic state labels."""
    old_to_new_trans_keys = {trans_key: i for i, (trans_key, _) in enumerate(sorted(fsm.alphabet.by_transition.items(), key=lambda x: sorted(x[1])))}
    new_symbol_mapping = {symbol: old_to_new_trans_keys[trans_key] for symbol, trans_key in fsm.alphabet._symbol_mapping.items()}
    new_alphabet = BetterAlphabet(new_symbol_mapping)
    new_map = {from_state: {old_to_new_trans_keys[trans_key]: to_state for trans_key, to_state in trans_map.items()} for from_state, trans_map in fsm.map.items()}
    old_to_new_states = {}
    old_to_new_states[fsm.initial] = 0
    i = 0
    seen = {fsm.initial}
    old_state_queue = [fsm.initial]
    while old_state_queue:
        old_state = old_state_queue.pop(-1)
        transitions = new_map[old_state]
        sorted_transitions = sorted(transitions.items(), key=lambda v: v[0])
        for _, old_state in sorted_transitions:
            if old_state not in seen:
                old_state_queue.append(old_state)
                seen.add(old_state)
            if old_state not in old_to_new_states:
                i += 1
                old_to_new_states[old_state] = i
    new_map = dict(sorted(((old_to_new_states[from_state], dict(sorted(((trans_key, old_to_new_states[to_state]) for trans_key, to_state in trans_map.items()), key=lambda v: v[0]))) for from_state, trans_map in new_map.items()), key=lambda v: v[0]))
    new_initial = 0
    new_finals = frozenset(sorted((old_to_new_states[old_state] for old_state in fsm.finals)))
    new_states = frozenset(sorted(new_map.keys()))
    new_fsm = BetterFSM(new_alphabet, new_states, new_initial, new_finals, new_map)
    return (new_fsm, old_to_new_states)