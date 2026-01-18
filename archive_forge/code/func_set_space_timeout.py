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
def set_space_timeout(space_id: str, hf_token: str | None=None, timeout_in_seconds: int=300):
    headers = huggingface_hub.utils.build_hf_headers(token=hf_token, library_name='gradio_client', library_version=__version__)
    try:
        httpx.post(f'https://huggingface.co/api/spaces/{space_id}/sleeptime', json={'seconds': timeout_in_seconds}, headers=headers)
    except httpx.HTTPStatusError as e:
        raise SpaceDuplicationError(f'Could not set sleep timeout on duplicated Space. Please visit {SPACE_URL.format(space_id)} to set a timeout manually to reduce billing charges.') from e