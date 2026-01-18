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
def scn_metadata(self) -> str | None:
    """Leica SCN XML metadata from ImageDescription tag."""
    if not self.is_scn:
        return None
    return self.pages.first.description