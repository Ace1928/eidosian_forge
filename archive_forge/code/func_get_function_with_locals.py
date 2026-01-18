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
def get_function_with_locals(fn: Callable, blocks: Blocks, event_id: str | None, in_event_listener: bool, request: Request | None):

    def before_fn(blocks, event_id):
        from gradio.context import LocalContext
        LocalContext.blocks.set(blocks)
        LocalContext.in_event_listener.set(in_event_listener)
        LocalContext.event_id.set(event_id)
        LocalContext.request.set(request)

    def after_fn():
        from gradio.context import LocalContext
        LocalContext.in_event_listener.set(False)
        LocalContext.request.set(None)
    return function_wrapper(fn, before_fn=before_fn, before_args=(blocks, event_id), after_fn=after_fn)