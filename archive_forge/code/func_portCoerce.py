from __future__ import annotations
import getopt
import inspect
import os
import sys
import textwrap
from os import path
from typing import Any, Dict, Optional, cast
from twisted.python import reflect, util
def portCoerce(value):
    """
    Coerce a string value to an int port number, and checks the validity.
    """
    value = int(value)
    if value < 0 or value > 65535:
        raise ValueError(f'Port number not in range: {value}')
    return value