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
def is_special_typed_parameter(name, parameter_types):
    from gradio.helpers import EventData
    from gradio.oauth import OAuthProfile, OAuthToken
    from gradio.routes import Request
    'Checks if parameter has a type hint designating it as a gr.Request, gr.EventData, gr.OAuthProfile or gr.OAuthToken.'
    hint = parameter_types.get(name)
    if not hint:
        return False
    is_request = hint == Request
    is_oauth_arg = hint in (OAuthProfile, Optional[OAuthProfile], OAuthToken, Optional[OAuthToken])
    is_event_data = inspect.isclass(hint) and issubclass(hint, EventData)
    return is_request or is_event_data or is_oauth_arg