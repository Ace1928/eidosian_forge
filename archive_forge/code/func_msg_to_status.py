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
@staticmethod
def msg_to_status(msg: str) -> Status:
    """Map the raw message from the backend to the status code presented to users."""
    return {ServerMessage.send_hash: Status.JOINING_QUEUE, ServerMessage.queue_full: Status.QUEUE_FULL, ServerMessage.estimation: Status.IN_QUEUE, ServerMessage.send_data: Status.SENDING_DATA, ServerMessage.process_starts: Status.PROCESSING, ServerMessage.process_generating: Status.ITERATING, ServerMessage.process_completed: Status.FINISHED, ServerMessage.progress: Status.PROGRESS, ServerMessage.log: Status.LOG, ServerMessage.server_stopped: Status.FINISHED}[msg]