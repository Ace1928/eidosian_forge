import contextlib
from typing import Dict, Iterator, Set, Union
import torch
from torch.cuda import _lazy_call
from torch.utils.checkpoint import detach_variable
from .initialize import get_data_parallel_rank, get_model_parallel_rank
def model_parallel_cuda_manual_seed(seed: int) -> None:
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    Two set of RNG states are tracked:
        default state: This is for data parallelism and is the same among a
                       set of model parallel GPUs but different across
                       different model paralle groups. This is used for
                       example for dropout in the non-model-parallel regions.
        model-parallel state: This state is different among a set of model
                              parallel GPUs, but the same across data parallel
                              groups. This is used for example for dropout in
                              model parallel regions.
    """
    offset = seed + 2718
    model_parallel_seed = offset + get_model_parallel_rank()
    data_parallel_seed = seed
    if torch.distributed.get_rank() == 0:
        print('> initializing model parallel cuda seeds on global rank {}, model parallel rank {}, and data parallel rank {} with model parallel seed: {} and data parallel seed: {}'.format(torch.distributed.get_rank(), get_model_parallel_rank(), get_data_parallel_rank(), model_parallel_seed, data_parallel_seed), flush=True)
    if torch.cuda.is_available():
        _CUDA_RNG_STATE_TRACKER.reset()
        torch.cuda.manual_seed(data_parallel_seed)
        _CUDA_RNG_STATE_TRACKER.add(_MODEL_PARALLEL_RNG_TRACKER_NAME, model_parallel_seed)