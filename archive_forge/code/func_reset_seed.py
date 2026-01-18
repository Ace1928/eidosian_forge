import logging
import os
import random
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state
from typing import Any, Dict, Optional
import numpy as np
import torch
from lightning_fabric.utilities.rank_zero import _get_rank, rank_prefixed_message, rank_zero_only, rank_zero_warn
def reset_seed() -> None:
    """Reset the seed to the value that :func:`~lightning_fabric.utilities.seed.seed_everything` previously set.

    If :func:`~lightning_fabric.utilities.seed.seed_everything` is unused, this function will do nothing.

    """
    seed = os.environ.get('PL_GLOBAL_SEED', None)
    if seed is None:
        return
    workers = os.environ.get('PL_SEED_WORKERS', '0')
    seed_everything(int(seed), workers=bool(int(workers)))