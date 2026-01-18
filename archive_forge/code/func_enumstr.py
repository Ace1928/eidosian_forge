from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
def enumstr(enum: Any, /) -> str:
    """Return short string representation of Enum member.

    >>> enumstr(PHOTOMETRIC.RGB)
    'RGB'

    """
    name = enum.name
    if name is None:
        name = str(enum)
    return name