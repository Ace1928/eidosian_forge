from __future__ import annotations
import base64
import os
import platform
import sys
from functools import reduce
from typing import Any
def fg(s: int) -> str:
    return COLOR_SEQ % s