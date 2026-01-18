import array
import asyncio
import atexit
from inspect import getfullargspec
import os
import re
import typing
import zlib
from typing import (
@classmethod
def configured_class(cls):
    """Returns the currently configured class."""
    base = cls.configurable_base()
    if base.__dict__.get('_Configurable__impl_class') is None:
        base.__impl_class = cls.configurable_default()
    if base.__impl_class is not None:
        return base.__impl_class
    else:
        raise ValueError('configured class not found')