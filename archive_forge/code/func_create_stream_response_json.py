import argparse
import asyncio
from http import HTTPStatus
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from typing import AsyncGenerator, Optional
from dataclasses import dataclass
import fastapi
from fastapi import BackgroundTasks, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
import uvicorn
import ray
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.outputs import RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from ochat.config import MODEL_CONFIG_MAP
from ochat.serving import openai_api_protocol, async_tokenizer
from transformers.utils.hub import cached_file
def create_stream_response_json(index: int, text: str, finish_reason: Optional[str]=None) -> str:
    choice_data = openai_api_protocol.ChatCompletionResponseStreamChoice(index=index, delta=openai_api_protocol.DeltaMessage(content=text), finish_reason=finish_reason)
    response = openai_api_protocol.ChatCompletionStreamResponse(id=request_id, choices=[choice_data], model=model_name)
    return response.json(exclude_unset=True, ensure_ascii=False)