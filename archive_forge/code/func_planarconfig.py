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
def planarconfig(self) -> int | None:
    """Value of PlanarConfiguration tag."""
    if self.separate_samples > 1:
        return 2
    if self.contig_samples > 1:
        return 1
    return None