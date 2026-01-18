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
def query_huggingface_inference_endpoints(*data):
    if preprocess is not None:
        data = preprocess(*data)
    data = fn(*data)
    if postprocess is not None:
        data = postprocess(data)
    return data