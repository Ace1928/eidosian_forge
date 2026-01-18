from typing import Dict, List, Optional, Tuple
import torch
import torch.distributed
from vllm.config import (CacheConfig, DeviceConfig, ModelConfig,
from vllm.model_executor import set_random_seed
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.sequence import SamplerOutput, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.model_runner import ModelRunner
@torch.inference_mode()
def execute_model(self, seq_group_metadata_list: Optional[List[SequenceGroupMetadata]]=None, blocks_to_swap_in: Optional[Dict[int, int]]=None, blocks_to_swap_out: Optional[Dict[int, int]]=None, blocks_to_copy: Optional[Dict[int, List[int]]]=None) -> Optional[SamplerOutput]:
    if self.is_driver_worker:
        assert seq_group_metadata_list is not None
        num_seq_groups = len(seq_group_metadata_list)
        assert blocks_to_swap_in is not None
        assert blocks_to_swap_out is not None
        assert blocks_to_copy is not None
        data = {'num_seq_groups': num_seq_groups, 'blocks_to_swap_in': blocks_to_swap_in, 'blocks_to_swap_out': blocks_to_swap_out, 'blocks_to_copy': blocks_to_copy}
        broadcast_tensor_dict(data, src=0)
    else:
        data = broadcast_tensor_dict(src=0)
        num_seq_groups = data['num_seq_groups']
        blocks_to_swap_in = data['blocks_to_swap_in']
        blocks_to_swap_out = data['blocks_to_swap_out']
        blocks_to_copy = data['blocks_to_copy']
    self.cache_swap(blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)
    if num_seq_groups == 0:
        return {}
    output = self.model_runner.execute_model(seq_group_metadata_list, self.gpu_cache)
    return output