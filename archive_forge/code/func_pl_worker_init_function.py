import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning_fabric.utilities.rank_zero import _get_rank, rank_prefixed_message, rank_zero_only, rank_zero_warn
def pl_worker_init_function(worker_id: int, rank: Optional[int]=None) -> None:
    """The worker_init_fn that Lightning automatically adds to your dataloader if you previously set the seed with
    ``seed_everything(seed, workers=True)``.

    See also the PyTorch documentation on
    `randomness in DataLoaders <https://pytorch.org/docs/stable/notes/randomness.html#dataloader>`_.

    """
    global_rank = rank if rank is not None else rank_zero_only.rank
    process_seed = torch.initial_seed()
    base_seed = process_seed - worker_id
    log.debug(f'Initializing random number generators of process {global_rank} worker {worker_id} with base seed {base_seed}')
    ss = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(ss.generate_state(4))
    torch_ss, stdlib_ss = ss.spawn(2)
    torch.manual_seed(torch_ss.generate_state(1, dtype=np.uint64)[0])
    stdlib_seed = (stdlib_ss.generate_state(2, dtype=np.uint64).astype(object) * [1 << 64, 1]).sum()
    random.seed(stdlib_seed)