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
def gdal_metadata(self) -> dict[str, Any] | None:
    """GDAL XML metadata from GDAL_METADATA tag."""
    if not self.is_gdal:
        return None
    return self.pages.first.tags.valueof(42112)