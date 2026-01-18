from __future__ import annotations
import io
import json
import os
import typing as t
from .encoding import (
def read_binary_file(path: str) -> bytes:
    """Return the contents of the specified path as bytes."""
    with open_binary_file(path) as file_obj:
        return file_obj.read()