from __future__ import annotations
import ast
import asyncio
import copy
import dataclasses
import functools
import importlib
import importlib.util
import inspect
import json
import json.decoder
import os
import pkgutil
import re
import sys
import tempfile
import threading
import time
import traceback
import typing
import urllib.parse
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from io import BytesIO
from numbers import Number
from pathlib import Path
from types import AsyncGeneratorType, GeneratorType, ModuleType
from typing import (
import anyio
import gradio_client.utils as client_utils
import httpx
from gradio_client.documentation import document
from typing_extensions import ParamSpec
import gradio
from gradio.context import Context
from gradio.data_classes import FileData
from gradio.strings import en
def get_extension_from_file_path_or_url(file_path_or_url: str) -> str:
    """
    Returns the file extension (without the dot) from a file path or URL. If the file path or URL does not have a file extension, returns an empty string.
    For example, "https://example.com/avatar/xxxx.mp4?se=2023-11-16T06:51:23Z&sp=r" would return "mp4".
    """
    parsed_url = urllib.parse.urlparse(file_path_or_url)
    file_extension = os.path.splitext(os.path.basename(parsed_url.path))[1]
    return file_extension[1:] if file_extension else ''