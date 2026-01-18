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
def astrotiff_metadata(self) -> dict[str, Any] | None:
    """AstroTIFF metadata from ImageDescription tag."""
    if not self.is_astrotiff:
        return None
    return astrotiff_description_metadata(self.pages.first.description)