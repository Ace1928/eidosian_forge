import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def get_max_num_running_seqs(self) -> int:
    """The maximum number of sequences running in parallel in the remaining
        lifetime of the request."""
    if self.sampling_params.use_beam_search:
        return self.sampling_params.best_of
    else:
        if self.sampling_params.best_of > self.num_seqs():
            return self.sampling_params.best_of
        return self.num_unfinished_seqs()