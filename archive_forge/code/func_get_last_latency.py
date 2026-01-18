import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def get_last_latency(self, now: float) -> float:
    """Gets last token latency for Request level timings."""
    latency = now - self.metrics.last_token_time
    self.metrics.last_token_time = now
    return latency