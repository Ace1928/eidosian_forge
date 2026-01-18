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
def is_bif(self) -> bool:
    """Page contains Ventana metadata."""
    try:
        return 700 in self.tags and ('Ventana' in self.software or self.software[:17] == 'ScanOutputManager' or self.description == 'Label Image' or (self.description == 'Label_Image') or (self.description == 'Probability_Image'))
    except Exception:
        return False