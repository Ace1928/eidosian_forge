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
def get_function_params(func: Callable) -> list[tuple[str, bool, Any]]:
    params_info = []
    signature = inspect.signature(func)
    for name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            break
        if parameter.default is inspect.Parameter.empty:
            params_info.append((name, False, None))
        else:
            params_info.append((name, True, parameter.default))
    return params_info