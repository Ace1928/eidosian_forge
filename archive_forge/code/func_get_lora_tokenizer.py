from typing import List, Optional, Tuple, Union
from transformers import (AutoTokenizer, PreTrainedTokenizer,
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.utils import make_async, LRUCache
from vllm.transformers_utils.tokenizers import *
def get_lora_tokenizer(self, lora_request: Optional[LoRARequest]) -> 'PreTrainedTokenizer':
    if not lora_request or not self.enable_lora:
        return self.tokenizer
    if lora_request.lora_int_id not in self.lora_tokenizers:
        tokenizer = get_lora_tokenizer(lora_request, **self.tokenizer_config) or self.tokenizer
        self.lora_tokenizers.put(lora_request.lora_int_id, tokenizer)
        return tokenizer
    else:
        return self.lora_tokenizers.get(lora_request.lora_int_id)