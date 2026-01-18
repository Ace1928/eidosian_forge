import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def get_next_fsm_states(fsms: List['Guide'], fsm_states: List[int], next_token_ids: torch.Tensor) -> List[int]:
    """

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    next_token_ids
        The tokens that were just generated.

    Returns
    -------
    A `torch.Tensor` object that represents the next logit mask.

    """
    return [fsm.get_next_state(fsm_state, int(token_id[0])) for fsm, fsm_state, token_id in zip(fsms, fsm_states, next_token_ids)]