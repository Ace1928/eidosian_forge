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
def get_cancel_function(dependencies: list[dict[str, Any]]) -> tuple[Callable, list[int]]:
    fn_to_comp = {}
    for dep in dependencies:
        if Context.root_block:
            fn_index = next((i for i, d in enumerate(Context.root_block.dependencies) if d == dep))
            fn_to_comp[fn_index] = [Context.root_block.blocks[o] for o in dep['outputs']]

    async def cancel(session_hash: str) -> None:
        task_ids = {f'{session_hash}_{fn}' for fn in fn_to_comp}
        await cancel_tasks(task_ids)
    return (cancel, list(fn_to_comp.keys()))