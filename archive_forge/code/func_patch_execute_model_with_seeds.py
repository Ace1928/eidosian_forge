import torch
from typing import List, Optional, Dict
from vllm.worker.worker import Worker
from vllm.utils import get_distributed_init_method, get_ip, get_open_port
from vllm.engine.arg_utils import EngineArgs
from vllm.sequence import SequenceGroupMetadata, SequenceData
from vllm.sampling_params import SamplingParams
from vllm.worker.cache_engine import CacheEngine
from vllm.model_executor.utils import set_random_seed
from dataclasses import dataclass, fields
def patch_execute_model_with_seeds(worker: Worker, rand_seeds: List[int]):
    seed_iter = iter(rand_seeds)
    original_execute_model = worker.execute_model

    def new_execute_model(*args, **kwargs):
        result = original_execute_model(*args, **kwargs)
        set_random_seed(next(seed_iter))
        return result
    return new_execute_model