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
def restore_session_state(app: App, body: PredictBody):
    event_id = body.event_id
    session_hash = getattr(body, 'session_hash', None)
    if session_hash is not None:
        session_state = app.state_holder[session_hash]
        if event_id is None:
            iterator = None
        elif event_id in app.iterators_to_reset:
            iterator = None
            app.iterators_to_reset.remove(event_id)
        else:
            iterator = app.iterators.get(event_id)
    else:
        session_state = SessionState(app.get_blocks())
        iterator = None
    return (session_state, iterator)