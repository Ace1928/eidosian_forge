from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
def supports_images() -> bool:
    return sys.stdin.isatty() and ITERM_PROFILE is not None