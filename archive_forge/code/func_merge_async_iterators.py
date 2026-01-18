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
def merge_async_iterators(*iterators):
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    queue = asyncio.Queue()
    finished = [False] * len(iterators)

    async def producer(i, iterator):
        async for item in iterator:
            await queue.put((i, item))
        finished[i] = True
    _tasks = [asyncio.create_task(producer(i, iterator)) for i, iterator in enumerate(iterators)]

    async def consumer():
        while not all(finished) or not queue.empty():
            item = await queue.get()
            yield item
        await asyncio.gather(*_tasks)
    return consumer()