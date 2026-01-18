import builtins
import codecs
import enum
import io
import json
import os
import types
import typing
from typing import (
import attr
def default_dumps(obj: Any) -> str:
    """
    Fake ``dumps()`` function to use as a default marker.
    """
    raise NotImplementedError