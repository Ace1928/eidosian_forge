from __future__ import annotations
import asyncio
import contextlib
import sys
import inspect
import json
import mimetypes
import os
import posixpath
import secrets
import threading
import time
import traceback
from pathlib import Path
from queue import Empty as EmptyQueue
from typing import (
import fastapi
import httpx
import markupsafe
import orjson
from fastapi import (
from fastapi.responses import (
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.templating import Jinja2Templates
from gradio_client import utils as client_utils
from gradio_client.documentation import document
from gradio_client.utils import ServerMessage
from jinja2.exceptions import TemplateNotFound
from multipart.multipart import parse_options_header
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.responses import RedirectResponse, StreamingResponse
import gradio
from gradio import ranged_response, route_utils, utils, wasm_utils
from gradio.context import Context
from gradio.data_classes import (
from gradio.exceptions import Error
from gradio.oauth import attach_oauth
from gradio.route_utils import (  # noqa: F401
from gradio.server_messages import (
from gradio.state_holder import StateHolder
from gradio.utils import get_package_version, get_upload_folder
def build_proxy_request(self, url_path):
    url = httpx.URL(url_path)
    assert self.blocks
    is_safe_url = any((url.host == httpx.URL(root).host for root in self.blocks.proxy_urls))
    if not is_safe_url:
        raise PermissionError('This URL cannot be proxied.')
    is_hf_url = url.host.endswith('.hf.space')
    headers = {}
    if Context.hf_token is not None and is_hf_url:
        headers['Authorization'] = f'Bearer {Context.hf_token}'
    rp_req = client.build_request('GET', url, headers=headers)
    return rp_req