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
def convert_langchain_message(message: ls_schemas.BaseMessageLike) -> dict:
    """Convert a LangChain message to an example."""
    converted: Dict[str, Any] = {'type': message.type, 'data': {'content': message.content}}
    if message.additional_kwargs and len(message.additional_kwargs) > 0:
        converted['data']['additional_kwargs'] = {**message.additional_kwargs}
    return converted