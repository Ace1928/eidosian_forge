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
def load_blocks_from_repo(name: str, src: str | None=None, hf_token: str | None=None, alias: str | None=None, **kwargs) -> Blocks:
    """Creates and returns a Blocks instance from a Hugging Face model or Space repo."""
    if src is None:
        tokens = name.split('/')
        if len(tokens) <= 1:
            raise ValueError('Either `src` parameter must be provided, or `name` must be formatted as {src}/{repo name}')
        src = tokens[0]
        name = '/'.join(tokens[1:])
    factory_methods: dict[str, Callable] = {'huggingface': from_model, 'models': from_model, 'spaces': from_spaces}
    if src.lower() not in factory_methods:
        raise ValueError(f'parameter: src must be one of {factory_methods.keys()}')
    if hf_token is not None:
        if Context.hf_token is not None and Context.hf_token != hf_token:
            warnings.warn('You are loading a model/Space with a different access token than the one you used to load a previous model/Space. This is not recommended, as it may cause unexpected behavior.')
        Context.hf_token = hf_token
    blocks: gradio.Blocks = factory_methods[src](name, hf_token, alias, **kwargs)
    return blocks