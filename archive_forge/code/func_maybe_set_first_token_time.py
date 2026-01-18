import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def maybe_set_first_token_time(self, time: float) -> None:
    """Sets the first token time for Request level timings."""
    if self.metrics.first_token_time is None:
        self.metrics.first_token_time = time