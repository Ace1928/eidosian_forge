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
def from_spaces_blocks(space: str, hf_token: str | None) -> Blocks:
    client = Client(space, hf_token=hf_token, upload_files=False, download_files=False, _skip_components=False)
    if client.app_version < version.Version('4.0.0b14'):
        raise GradioVersionIncompatibleError(f'Gradio version 4.x cannot load spaces with versions less than 4.x ({client.app_version}).Please downgrade to version 3 to load this space.')
    predict_fns = []
    for fn_index, endpoint in enumerate(client.endpoints):
        if not isinstance(endpoint, Endpoint):
            raise TypeError(f'Expected endpoint to be an Endpoint, but got {type(endpoint)}')
        helper = client.new_helper(fn_index)
        if endpoint.backend_fn:
            predict_fns.append(endpoint.make_end_to_end_fn(helper))
        else:
            predict_fns.append(None)
    return gradio.Blocks.from_config(client.config, predict_fns, client.src)