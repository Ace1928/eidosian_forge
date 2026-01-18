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
def is_file_obj_with_url(d) -> bool:
    """
    Check if the given value is a valid FileData object dictionary in newer versions of Gradio
    where the file objects include a specific "meta" key, and ALSO include a "url" key, e.g.
    {
        "path": "path/to/file",
        "url": "/file=path/to/file",
        "meta": {"_type: "gradio.FileData"}
    }
    """
    return is_file_obj_with_meta(d) and 'url' in d and isinstance(d['url'], str)