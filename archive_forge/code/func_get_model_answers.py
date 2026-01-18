from typing import Optional
import argparse
import os
import asyncio
from glob import glob
import orjson
import openai
from tqdm import tqdm
from openai.error import RateLimitError, ServiceUnavailableError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from vllm import LLM, SamplingParams
from transformers.utils.hub import cached_file
from ochat.evaluation.match_answer import MATCH_ANSWER_FUNCTION
from ochat.config import MODEL_CONFIG_MAP
def get_model_answers(model: str, questions: list, condition: str, system_msg: str, model_type: str, tensor_parallel_size: int):
    if model_type is None:
        with open(cached_file(path_or_repo_id=model, filename='openchat.json'), 'r') as f:
            model_type = orjson.loads(f.read())['model_type']
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)
    engine = LLM(model, max_num_batched_tokens=model_config.model_max_context, max_model_len=model_config.model_max_context, tensor_parallel_size=tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0, max_tokens=None, stop_token_ids=conv_template.eot_tokens_, ignore_eos=True)
    prompts, prompt_indices = tokenize_questions(model_config, conv_template, questions, condition=condition, system_msg=system_msg)
    responses = engine.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
    for idx, resp in zip(prompt_indices, responses):
        questions[idx]['response'] = _strip_first_space(resp.outputs[0].text)
    return questions