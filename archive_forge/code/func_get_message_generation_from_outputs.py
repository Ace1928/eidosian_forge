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
def get_message_generation_from_outputs(outputs: Mapping[str, Any]) -> Dict[str, Any]:
    """Retrieve the message generation from the given outputs.

    Args:
        outputs (Mapping[str, Any]): The outputs dictionary.

    Returns:
        Dict[str, Any]: The message generation.

    Raises:
        ValueError: If no generations are found or if multiple generations are present.
    """
    if 'generations' not in outputs:
        raise ValueError(f'No generations found in in run with output: {outputs}.')
    generations = outputs['generations']
    if len(generations) != 1:
        raise ValueError(f'Chat examples expect exactly one generation. Found {len(generations)} generations: {generations}.')
    first_generation = generations[0]
    if 'message' not in first_generation:
        raise ValueError(f'Unexpected format for generation: {first_generation}. Generation does not have a message.')
    return _convert_message(first_generation['message'])