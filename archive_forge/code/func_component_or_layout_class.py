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
def component_or_layout_class(cls_name: str) -> type[Component] | type[BlockContext]:
    """
    Returns the component, template, or layout class with the given class name, or
    raises a ValueError if not found.

    Parameters:
    cls_name (str): lower-case string class name of a component
    Returns:
    cls: the component class
    """
    import gradio.blocks
    import gradio.components
    import gradio.layouts
    import gradio.templates
    components = [(name, cls) for name, cls in gradio.components.__dict__.items() if isinstance(cls, type)]
    templates = [(name, cls) for name, cls in gradio.templates.__dict__.items() if isinstance(cls, type)]
    layouts = [(name, cls) for name, cls in gradio.layouts.__dict__.items() if isinstance(cls, type)]
    for name, cls in components + templates + layouts:
        if name.lower() == cls_name.replace('_', '') and (issubclass(cls, gradio.components.Component) or issubclass(cls, gradio.blocks.BlockContext)):
            return cls
    raise ValueError(f'No such component or layout: {cls_name}')