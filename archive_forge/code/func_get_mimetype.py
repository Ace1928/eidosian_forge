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
def get_mimetype(filename: str) -> str | None:
    if filename.endswith('.vtt'):
        return 'text/vtt'
    mimetype = mimetypes.guess_type(filename)[0]
    if mimetype is not None:
        mimetype = mimetype.replace('x-wav', 'wav').replace('x-flac', 'flac')
    return mimetype