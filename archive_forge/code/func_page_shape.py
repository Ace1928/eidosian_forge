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
def page_shape(self) -> tuple[int, int, int, int, int]:
    """Normalized 5D shape of image array in single page."""
    return (self.separate_samples, self.depth, self.length, self.width, self.contig_samples)