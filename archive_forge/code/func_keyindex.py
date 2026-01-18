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
def keyindex(key: str, /) -> tuple[str, int]:
    index = 0
    i = len(key.rstrip('0123456789'))
    if i < len(key):
        index = int(key[i:]) - 1
        key = key[:i]
    return (key, index)