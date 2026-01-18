import dataclasses
import math
from typing import TYPE_CHECKING, Callable, Iterator, List, Optional, Tuple
import torch
def is_generation_finished(fsms: List['Guide'], fsm_states: List[int]) -> bool:
    """Determine if the generation is finished.

    A generation is considered finished if the FSM of every sequence in the
    batch is in a final state.

    A better solution is to return finished sequences as soon as their FSM
    is in a final state.

    Parameters
    ----------
    fsm
        The finite-state machine used to monitor this batch.
    fsm_states
        The FSM states corresponding to each sequence in the batch.

    Returns
    -------
    Whether all sequences are finished sampling.

    """
    return all([fsm.is_final_state(state) for fsm, state in zip(fsms, fsm_states)])