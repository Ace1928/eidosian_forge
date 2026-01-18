import pytest
import ray
import vllm
from vllm.lora.request import LoRARequest
from .conftest import cleanup
@ray.remote(num_gpus=1)
def get_num_gpu_blocks_lora():
    llm = vllm.LLM(MODEL_PATH, enable_lora=True, max_num_seqs=16)
    num_gpu_blocks_lora_warmup = llm.llm_engine.cache_config.num_gpu_blocks
    return num_gpu_blocks_lora_warmup