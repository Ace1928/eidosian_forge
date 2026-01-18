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
@cached_property
def gdal_structural_metadata(self) -> dict[str, Any] | None:
    """Non-TIFF GDAL structural metadata."""
    return read_gdal_structural_metadata(self._fh)