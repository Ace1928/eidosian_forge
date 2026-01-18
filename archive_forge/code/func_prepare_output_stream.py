import asyncio
import json
import warnings
from abc import ABC
from typing import (
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.pydantic_v1 import BaseModel, Extra, Field, root_validator
from langchain_core.utils import get_from_dict_or_env
from langchain_community.llms.utils import enforce_stop_tokens
from langchain_community.utilities.anthropic import (
@classmethod
def prepare_output_stream(cls, provider: str, response: Any, stop: Optional[List[str]]=None, messages_api: bool=False) -> Iterator[GenerationChunk]:
    stream = response.get('body')
    if not stream:
        return
    if messages_api:
        output_key = 'message'
    else:
        output_key = cls.provider_to_output_key_map.get(provider, '')
    if not output_key:
        raise ValueError(f'Unknown streaming response output key for provider: {provider}')
    for event in stream:
        chunk = event.get('chunk')
        if not chunk:
            continue
        chunk_obj = json.loads(chunk.get('bytes').decode())
        if provider == 'cohere' and (chunk_obj['is_finished'] or chunk_obj[output_key] == '<EOS_TOKEN>'):
            return
        elif provider == 'mistral' and chunk_obj.get(output_key, [{}])[0].get('stop_reason', '') == 'stop':
            return
        elif messages_api and chunk_obj.get('type') == 'content_block_stop':
            return
        if messages_api and chunk_obj.get('type') in ('message_start', 'content_block_start', 'content_block_delta'):
            if chunk_obj.get('type') == 'content_block_delta':
                chk = _stream_response_to_generation_chunk(chunk_obj)
                yield chk
            else:
                continue
        else:
            yield GenerationChunk(text=chunk_obj[output_key] if provider != 'mistral' else chunk_obj[output_key][0]['text'], generation_info={GUARDRAILS_BODY_KEY: chunk_obj.get(GUARDRAILS_BODY_KEY) if GUARDRAILS_BODY_KEY in chunk_obj else None})