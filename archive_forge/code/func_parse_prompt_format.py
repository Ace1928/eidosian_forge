import asyncio
import time
from fastapi import Request
from typing import AsyncGenerator, AsyncIterator, Callable, List, Optional, Dict, Tuple
from vllm.logger import init_logger
from vllm.utils import random_uuid
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import OpenAIServing, LoRA
from vllm.model_executor.guided_decoding import get_guided_decoding_logits_processor
def parse_prompt_format(prompt) -> Tuple[bool, list]:
    prompt_is_tokens = False
    prompts = [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError('please provide at least one prompt')
        elif isinstance(prompt[0], str):
            prompt_is_tokens = False
            prompts = prompt
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]
        elif isinstance(prompt[0], list) and isinstance(prompt[0][0], int):
            prompt_is_tokens = True
            prompts = prompt
        else:
            raise ValueError('prompt must be a string, array of strings, array of tokens, or array of token arrays')
    return (prompt_is_tokens, prompts)