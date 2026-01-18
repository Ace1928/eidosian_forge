from itertools import product
import math
import pytest
import torch
import transformers
from transformers import (
from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter
def get_prompt_for_generation_eval(text, add_roles=True):
    description = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
    if add_roles:
        prompt = f'{description} ### Human: {text} ### Assistant:'
    else:
        prompt = f'{description} {text}'
    return prompt