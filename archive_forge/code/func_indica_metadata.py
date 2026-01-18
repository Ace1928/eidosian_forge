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
def indica_metadata(self) -> str | None:
    """IndicaLabs XML metadata from ImageDescription tag."""
    if not self.is_indica:
        return None
    return self.pages.first.description