from collections import namedtuple
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, Generator, List, Sequence, Set, Tuple
import numba
import numpy as np
from interegular.fsm import FSM, Alphabet, OblivionError, anything_else
from numba.typed.typedobjectutils import _nonoptional
def create_fsm_index_tokenizer(fsm: BetterFSM, tokenizer: 'Tokenizer') -> Tuple[Dict[int, Dict[int, int]], Set[int]]:
    """Construct an FMS index from a tokenizer.

    This uses the end-to-end approach of `create_fsm_index_end_to_end`.

    .. warning::

        `fsm` needs to be deterministically ordered so that future caching makes sense.

    """
    vocabulary, empty_token_ids = reduced_vocabulary(tokenizer)
    states_to_token_subsets = create_fsm_index_end_to_end(fsm.fsm_info, vocabulary)
    for state in fsm.fsm_info.finals:
        subset = states_to_token_subsets.get(state)
        if subset is not None:
            subset.add((tokenizer.eos_token_id, state))
    states_to_token_subsets = {k: dict(v) for k, v in states_to_token_subsets.items()}
    return (states_to_token_subsets, empty_token_ids)