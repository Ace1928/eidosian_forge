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
def download_tmp_copy_of_file(url_path: str, hf_token: str | None=None, dir: str | None=None) -> str:
    """Kept for backwards compatibility for 3.x spaces."""
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    headers = {'Authorization': 'Bearer ' + hf_token} if hf_token else {}
    directory = Path(dir or tempfile.gettempdir()) / secrets.token_hex(20)
    directory.mkdir(exist_ok=True, parents=True)
    file_path = directory / Path(url_path).name
    with httpx.stream('GET', url_path, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        with open(file_path, 'wb') as f:
            for chunk in response.iter_raw():
                f.write(chunk)
    return str(file_path.resolve())