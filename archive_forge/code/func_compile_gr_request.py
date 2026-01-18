from __future__ import annotations
import asyncio
import functools
import hashlib
import hmac
import json
import os
import re
import shutil
import sys
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass as python_dataclass
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile, _TemporaryFileWrapper
from typing import (
from urllib.parse import urlparse
import anyio
import fastapi
import gradio_client.utils as client_utils
import httpx
import multipart
from gradio_client.documentation import document
from multipart.multipart import parse_options_header
from starlette.datastructures import FormData, Headers, MutableHeaders, UploadFile
from starlette.formparsers import MultiPartException, MultipartPart
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from gradio import processing_utils, utils
from gradio.data_classes import PredictBody
from gradio.exceptions import Error
from gradio.helpers import EventData
from gradio.state_holder import SessionState
def compile_gr_request(app: App, body: PredictBody, fn_index_inferred: int, username: Optional[str], request: Optional[fastapi.Request]):
    if app.get_blocks().dependencies[fn_index_inferred]['cancels']:
        body.data = [body.session_hash]
    if body.request:
        if body.batched:
            gr_request = [Request(username=username, request=request)]
        else:
            gr_request = Request(username=username, request=body.request)
    else:
        if request is None:
            raise ValueError('request must be provided if body.request is None')
        gr_request = Request(username=username, request=request)
    return gr_request