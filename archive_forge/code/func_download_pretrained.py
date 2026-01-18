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
def download_pretrained(repo_id: str, revision: Optional[str]=None, repo_type: Optional[str]=None, auth_token: Optional[Union[str, bool]]=True, classifications: Optional[List[str]]=None, cache_dir: Optional[Union[str, Path]]=None, allow_patterns: Optional[Union[List[str], str]]=None, **kwargs):
    """
    Handles the downloading of pretrained models from HuggingFace.
    """
    if classifications is None:
        classifications = ['pytorch', 'tokenizer', 'config']
    if allow_patterns is None:
        allow_patterns = []
    if isinstance(allow_patterns, str):
        allow_patterns = [allow_patterns]
    for classification in classifications:
        allow_patterns.extend(reg_patterns[classification])
    if auth_token and auth_token is True:
        auth_token = get_huggingface_token(False)
    return snapshot_download(repo_id=repo_id, revision=revision, repo_type=repo_type, cache_dir=cache_dir, allow_patterns=allow_patterns, token=auth_token, **kwargs)