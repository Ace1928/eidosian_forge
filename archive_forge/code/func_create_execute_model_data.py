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
def create_execute_model_data(seq_group_metadata_list: List[SequenceGroupMetadata], blocks_to_swap_in: Optional[Dict[int, int]]=None, blocks_to_swap_out: Optional[Dict[int, int]]=None, blocks_to_copy: Optional[Dict[int, int]]=None) -> ExecuteModelData:
    if blocks_to_swap_in is None:
        blocks_to_swap_in = {}
    if blocks_to_swap_out is None:
        blocks_to_swap_out = {}
    if blocks_to_copy is None:
        blocks_to_copy = {}
    return ExecuteModelData(seq_group_metadata_list=seq_group_metadata_list, blocks_to_swap_in=blocks_to_swap_in, blocks_to_swap_out=blocks_to_swap_out, blocks_to_copy=blocks_to_copy)