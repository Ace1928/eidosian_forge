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
def eer_metadata(self) -> str | None:
    """EER AcquisitionMetadata XML from tag 65001."""
    if not self.is_eer:
        return None
    value = self.pages.first.tags.valueof(65001)
    return None if value is None else value.decode()