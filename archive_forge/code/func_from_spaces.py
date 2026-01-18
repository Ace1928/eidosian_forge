from __future__ import annotations
import json
import os
import re
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Callable
import httpx
import huggingface_hub
from gradio_client import Client
from gradio_client.client import Endpoint
from gradio_client.documentation import document
from packaging import version
import gradio
from gradio import components, external_utils, utils
from gradio.context import Context
from gradio.exceptions import (
from gradio.processing_utils import save_base64_to_cache, to_binary
def from_spaces(space_name: str, hf_token: str | None, alias: str | None, **kwargs) -> Blocks:
    space_url = f'https://huggingface.co/spaces/{space_name}'
    print(f'Fetching Space from: {space_url}')
    headers = {}
    if hf_token is not None:
        headers['Authorization'] = f'Bearer {hf_token}'
    iframe_url = httpx.get(f'https://huggingface.co/api/spaces/{space_name}/host', headers=headers).json().get('host')
    if iframe_url is None:
        raise ValueError(f'Could not find Space: {space_name}. If it is a private or gated Space, please provide your Hugging Face access token (https://huggingface.co/settings/tokens) as the argument for the `hf_token` parameter.')
    r = httpx.get(iframe_url, headers=headers)
    result = re.search('window.gradio_config = (.*?);[\\s]*</script>', r.text)
    try:
        config = json.loads(result.group(1))
    except AttributeError as ae:
        raise ValueError(f'Could not load the Space: {space_name}') from ae
    if 'allow_flagging' in config:
        return from_spaces_interface(space_name, config, alias, hf_token, iframe_url, **kwargs)
    else:
        if kwargs:
            warnings.warn('You cannot override parameters for this Space by passing in kwargs. Instead, please load the Space as a function and use it to create a Blocks or Interface locally. You may find this Guide helpful: https://gradio.app/using_blocks_like_functions/')
        return from_spaces_blocks(space=space_name, hf_token=hf_token)