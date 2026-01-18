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
@property
def sis_metadata(self) -> dict[str, Any] | None:
    """Olympus SIS metadata from OlympusSIS and OlympusINI tags."""
    if not self.is_sis:
        return None
    tags = self.pages.first.tags
    result = {}
    try:
        result.update(tags.valueof(33471))
    except Exception:
        pass
    try:
        result.update(tags.valueof(33560))
    except Exception:
        pass
    return result