import hashlib
import shutil
from pathlib import Path
from datetime import datetime
import asyncio
import aiofiles
import logging
from typing import Dict, List, Tuple, Union, Callable, Coroutine, Any, Optional
from functools import wraps
import threading
import ctypes
import sys
from PyQt5.QtWidgets import (
from PyQt5.QtCore import QDir, QThread, QObject, pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow
import os
import ctypes
def run_asyncio_loop(loop: asyncio.AbstractEventLoop, coro: Coroutine) -> None:
    """
    Runs the provided coroutine in the specified asyncio event loop.

    Args:
        loop (asyncio.AbstractEventLoop): The asyncio event loop to use.
        coro (Coroutine): The coroutine to run in the event loop.
    """
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(coro)
    except Exception as e:
        logger.error(f'Error running asyncio loop: {e}')