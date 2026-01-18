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
def get_messages_from_inputs(inputs: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Extract messages from the given inputs dictionary.

    Args:
        inputs (Mapping[str, Any]): The inputs dictionary.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing
            the extracted messages.

    Raises:
        ValueError: If no message(s) are found in the inputs dictionary.
    """
    if 'messages' in inputs:
        return [_convert_message(message) for message in inputs['messages']]
    if 'message' in inputs:
        return [_convert_message(inputs['message'])]
    raise ValueError(f'Could not find message(s) in run with inputs {inputs}.')