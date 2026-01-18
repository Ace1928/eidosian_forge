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
def repr_for_fp(fp: typing.IO[Any]) -> str:
    """
    Helper to make a useful repr() for a file-like object.
    """
    name = getattr(fp, 'name', None)
    if name is not None:
        return repr(name)
    else:
        return repr(fp)