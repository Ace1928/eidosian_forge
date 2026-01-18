from __future__ import annotations
import asyncio
import json
import os
import threading
import urllib.parse
import warnings
from typing import Any
import httpx
from packaging.version import Version
import gradio
from gradio import wasm_utils
from gradio.context import Context
from gradio.utils import get_package_version
def error_analytics(message: str) -> None:
    """
    Send error analytics if there is network
    Parameters:
        message: Details about error
    """
    if not analytics_enabled():
        return
    data = {'error': message}
    _do_analytics_request(url=f'{ANALYTICS_URL}gradio-error-analytics/', data=data)