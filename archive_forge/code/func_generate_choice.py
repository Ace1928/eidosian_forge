from functools import singledispatch
from typing import Callable, List
from outlines.generate.api import SequenceGenerator
from outlines.models import OpenAI
from outlines.samplers import Sampler, multinomial
from .regex import regex
def generate_choice(prompt: str, max_tokens: int=1):
    return model.generate_choice(prompt, choices, max_tokens)