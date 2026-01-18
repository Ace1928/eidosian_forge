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
def download_byte_stream(url: str, hf_token=None):
    arr = bytearray()
    headers = {'Authorization': 'Bearer ' + hf_token} if hf_token else {}
    with httpx.stream('GET', url, headers=headers) as r:
        for data in r.iter_bytes():
            arr += data
            yield data
    yield arr