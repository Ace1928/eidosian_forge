import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
@property
def prompt_token_ids(self) -> List[int]:
    return next(iter(self.seqs_dict.values())).data.prompt_token_ids