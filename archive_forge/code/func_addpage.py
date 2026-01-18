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
def addpage(page: TiffPage | TiffFrame, /) -> None:
    if not page.shape:
        return
    key = page.hash
    if key in seriesdict:
        for p in seriesdict[key]:
            if p.offset == page.offset:
                break
        else:
            seriesdict[key].append(page)
    else:
        keys.append(key)
        seriesdict[key] = [page]