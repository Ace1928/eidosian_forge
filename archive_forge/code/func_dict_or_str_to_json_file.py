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
def dict_or_str_to_json_file(jsn: str | dict | list, dir: str | Path | None=None):
    if dir is not None:
        os.makedirs(dir, exist_ok=True)
    file_obj = tempfile.NamedTemporaryFile(delete=False, suffix='.json', dir=dir, mode='w+')
    if isinstance(jsn, str):
        jsn = json.loads(jsn)
    json.dump(jsn, file_obj)
    file_obj.flush()
    return file_obj