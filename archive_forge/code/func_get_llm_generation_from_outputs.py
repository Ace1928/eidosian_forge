import contextlib
import enum
import functools
import logging
import os
import pathlib
import subprocess
import threading
from typing import (
import requests
from urllib3.util import Retry
from langsmith import schemas as ls_schemas
def get_llm_generation_from_outputs(outputs: Mapping[str, Any]) -> str:
    """Get the LLM generation from the outputs."""
    if 'generations' not in outputs:
        raise ValueError(f'No generations found in in run with output: {outputs}.')
    generations = outputs['generations']
    if len(generations) != 1:
        raise ValueError(f'Multiple generations in run: {generations}')
    first_generation = generations[0]
    if 'text' not in first_generation:
        raise ValueError(f'No text in generation: {first_generation}')
    return first_generation['text']