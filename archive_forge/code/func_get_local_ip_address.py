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
def get_local_ip_address() -> str:
    """
    Gets the public IP address or returns the string "No internet connection" if unable
    to obtain it or the string "Analytics disabled" if a user has disabled analytics.
    Does not make a new request if the IP address has already been obtained in the
    same Python session.
    """
    if not analytics_enabled():
        return 'Analytics disabled'
    if Context.ip_address is None:
        try:
            ip_address = httpx.get('https://checkip.amazonaws.com/', timeout=3).text.strip()
        except (httpx.ConnectError, httpx.ReadTimeout):
            ip_address = 'No internet connection'
        Context.ip_address = ip_address
    else:
        ip_address = Context.ip_address
    return ip_address