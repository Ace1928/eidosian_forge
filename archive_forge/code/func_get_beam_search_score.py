import copy
import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from vllm.block import LogicalTokenBlock
from vllm.prefix import Prefix
from vllm.sampling_params import SamplingParams
from vllm.lora.request import LoRARequest
def get_beam_search_score(self, length_penalty: float=1.0, seq_len: Optional[int]=None, eos_token_id: Optional[int]=None) -> float:
    """Calculate the beam search score with length penalty.

        Adapted from

        https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/generation/beam_search.py#L938
        """
    if seq_len is None:
        seq_len = self.get_len()
        if eos_token_id is not None and self.get_last_token_id() == eos_token_id:
            seq_len -= 1
    return self.get_cumulative_logprob() / seq_len ** length_penalty