from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
@numba.njit(cache=True)
def create_fsm_info(py_initial, py_finals, flat_transition_map_items, trans_key_to_states_items, py_anything_value, alphabet_symbol_mapping_items):
    trans_key_to_states = numba.typed.Dict.empty(numba.int64, nb_int_list_type)
    for trans_key_and_state in trans_key_to_states_items:
        trans_key_to_states.setdefault(trans_key_and_state[0], numba.typed.List.empty_list(numba.int64)).append(trans_key_and_state[1])
    flat_transition_map = numba.typed.Dict.empty(nb_int_pair_type, numba.int64)
    for trans_key_and_state in flat_transition_map_items:
        flat_transition_map[trans_key_and_state[0], trans_key_and_state[1]] = trans_key_and_state[2]
    alphabet_symbol_map = numba.typed.Dict.empty(nb_unichar_1_type, numba.int64)
    for symbol_and_trans_key in alphabet_symbol_mapping_items:
        alphabet_symbol_map[symbol_and_trans_key[0]] = symbol_and_trans_key[1]
    initial = numba.int64(py_initial)
    finals = set()
    for final in py_finals:
        finals.add(final)
    anything_value = numba.int64(py_anything_value)
    return FSMInfo(initial, finals, flat_transition_map, trans_key_to_states, anything_value, alphabet_symbol_map)