import asyncio
import concurrent.futures
import os
import subprocess
import sys
import tempfile
from importlib import util as importlib_util
from traitlets import Bool, default
from .html import HTMLExporter
def run_coroutine(coro):
    """Run an internal coroutine."""
    loop = asyncio.ProactorEventLoop() if IS_WINDOWS else asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)