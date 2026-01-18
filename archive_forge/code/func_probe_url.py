from __future__ import annotations
import asyncio
import base64
import copy
import json
import mimetypes
import os
import pkgutil
import secrets
import shutil
import tempfile
import warnings
from concurrent.futures import CancelledError
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, Optional, TypedDict
import fsspec.asyn
import httpx
import huggingface_hub
from huggingface_hub import SpaceStage
from websockets.legacy.protocol import WebSocketCommonProtocol
def probe_url(possible_url: str) -> bool:
    """
    Probe the given URL to see if it responds with a 200 status code (to HEAD, then to GET).
    """
    headers = {'User-Agent': 'gradio (https://gradio.app/; gradio-team@huggingface.co)'}
    try:
        with httpx.Client() as client:
            head_request = client.head(possible_url, headers=headers)
            if head_request.status_code == 405:
                return client.get(possible_url, headers=headers).is_success
            return head_request.is_success
    except Exception:
        return False