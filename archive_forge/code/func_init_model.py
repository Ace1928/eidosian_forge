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
def init_model(self) -> None:
    _init_distributed_environment(self.parallel_config, self.rank, self.distributed_init_method, distributed_backend='gloo')
    set_random_seed(self.model_config.seed)