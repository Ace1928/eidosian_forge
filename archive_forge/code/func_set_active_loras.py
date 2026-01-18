import contextlib
import time
from typing import Dict, List, Optional, Tuple, Set, Union
import numpy as np
import torch
import torch.nn as nn
from vllm.config import (DeviceConfig, ModelConfig, LoRAConfig, ParallelConfig,
from vllm.logger import init_logger
from vllm.model_executor import get_model, InputMetadata, SamplingMetadata
from vllm.model_executor.parallel_utils import cupy_utils
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils import custom_all_reduce
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.utils import in_wsl
def set_active_loras(self, lora_requests: List[LoRARequest], lora_mapping: LoRAMapping) -> None:
    if not self.lora_manager:
        raise RuntimeError('LoRA is not enabled.')
    self.lora_manager.set_active_loras(lora_requests, lora_mapping)