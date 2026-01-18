import os
import re
import base64
import requests
import json
import functools
import contextlib
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict, Any, TYPE_CHECKING
from lazyops.utils.logs import logger
from lazyops.types import BaseModel, lazyproperty, Literal
from pydantic.types import ByteSize
def get_huggingface_token(raise_error: Optional[bool]=False) -> Optional[str]:
    """
    Gets the HuggingFace API token
    """
    for key in {'HF_TOKEN', 'HUGGINGFACE_TOKEN', 'HUGGINGFACE_API_TOKEN'}:
        if key in os.environ:
            return os.environ[key]
    token_path = user_home.joinpath('.huggingface/token')
    if token_path.exists():
        with open(token_path, 'r') as f:
            return f.read().strip()
    if raise_error:
        raise ValueError('HuggingFace API token not found. Please run `huggingface-cli login` to login.')
    return None