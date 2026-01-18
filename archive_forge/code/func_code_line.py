from __future__ import annotations
import enum
import os
import io
import sys
import time
import platform
import shlex
import subprocess
import shutil
import typing as T
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
def code_line(text: str, line: str, colno: int) -> str:
    """Print a line with a caret pointing to the colno

    :param text: A message to display before the line
    :param line: The line of code to be pointed to
    :param colno: The column number to point at
    :return: A formatted string of the text, line, and a caret
    """
    return f'{text}\n{line}\n{' ' * colno}^'